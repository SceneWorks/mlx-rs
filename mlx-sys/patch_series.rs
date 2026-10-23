//! Native patch transaction: exact source/ordered-input receipts, never marker-only skips.
use std::collections::BTreeSet;

fn quoted(path: &str) -> String {
    assert!(
        !path.is_empty()
            && path
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"/._-".contains(&b))
    );
    assert!(!path.starts_with('/') && !path.split('/').any(|part| part == ".."));
    format!("'{path}'")
}

// Put each native input identity in its own Cargo-owned FetchContent source AND
// binary directory. An upgrade never reuses/resets an old patched checkout.
pub fn cmake_declaration(key: &str) -> String {
    assert!(key.bytes().all(|b| b.is_ascii_hexdigit()) && !key.is_empty());
    format!("GIT_TAG v0.32.0\n    SOURCE_DIR ${{CMAKE_CURRENT_BINARY_DIR}}/_deps/pmetal-{key}/mlx-src\n    BINARY_DIR ${{CMAKE_CURRENT_BINARY_DIR}}/_deps/pmetal-{key}/mlx-build\n    PATCH_COMMAND sh ${{CMAKE_CURRENT_SOURCE_DIR}}/patches/apply_patches.sh)")
}

fn patch_paths(patch: &str) -> BTreeSet<String> {
    fn path(header: &str, prefix: &str) -> Option<String> {
        if header == "/dev/null" {
            return None;
        }
        let path = header
            .strip_prefix(prefix)
            .expect("unsupported native patch path/header");
        quoted(path);
        Some(path.to_owned())
    }
    fn count(range: &str, sign: char) -> usize {
        let range = range
            .strip_prefix(sign)
            .expect("unsupported native patch hunk range");
        let parts: Vec<_> = range.split(',').collect();
        assert!(parts.len() <= 2, "unsupported native patch hunk range");
        parts[0].parse::<usize>().expect("native patch hunk start");
        if parts.len() == 2 {
            parts[1].parse().expect("native patch hunk count")
        } else {
            1
        }
    }
    let lines: Vec<_> = patch.lines().collect();
    let mut files = BTreeSet::new();
    let mut git_paths: Option<(String, String)> = None;
    let mut old: Option<Option<String>> = None;
    let mut active = false;
    let mut hunks = 0;
    let mail = lines.first().is_some_and(|line| line.starts_with("From "));
    let mut i = 0;
    if mail {
        let envelope = lines[0].strip_prefix("From ").unwrap();
        let hash = envelope
            .strip_suffix(" Mon Sep 17 00:00:00 2001")
            .expect("unsupported native patch mail envelope");
        assert!(
            hash.len() == 40 && hash.bytes().all(|c| c.is_ascii_hexdigit()),
            "unsupported native patch mail hash"
        );
        i = 1;
    }
    let mut footer = false;
    let mut footer_version = false;
    while i < lines.len() {
        let line = lines[i];
        if let Some(paths) = line.strip_prefix("diff --git ") {
            assert!(
                old.is_none() && git_paths.is_none() && (!active || hunks > 0),
                "incomplete native patch section"
            );
            let parts: Vec<_> = paths.split(' ').collect();
            assert_eq!(parts.len(), 2, "unsupported native patch diff paths");
            let a = path(parts[0], "a/").expect("native patch git old path");
            let b = path(parts[1], "b/").expect("native patch git new path");
            assert_eq!(a, b, "unsupported native patch rename/copy");
            git_paths = Some((a, b));
            active = false;
            hunks = 0;
        } else if let Some(header) = line.strip_prefix("--- ") {
            assert!(
                old.is_none() && (!active || hunks > 0),
                "incomplete native patch old header"
            );
            old = Some(path(header, "a/"));
            active = false;
            hunks = 0;
        } else if let Some(header) = line.strip_prefix("+++ ") {
            let old_path = old
                .take()
                .expect("native patch new header without old header");
            let new_path = path(header, "b/");
            assert!(
                old_path.is_some() || new_path.is_some(),
                "native patch has no file paths"
            );
            if let Some((a, b)) = git_paths.take() {
                assert!(
                    old_path.as_ref().is_none_or(|p| p == &a),
                    "native patch git/old header mismatch"
                );
                assert!(
                    new_path.as_ref().is_none_or(|p| p == &b),
                    "native patch git/new header mismatch"
                );
            }
            if let Some(p) = old_path {
                files.insert(p);
            }
            if let Some(p) = new_path {
                files.insert(p);
            }
            active = true;
        } else if line.starts_with("@@ ") {
            assert!(
                active && old.is_none(),
                "native patch hunk without file headers"
            );
            let parts: Vec<_> = line.split_whitespace().collect();
            assert!(
                parts.len() >= 4 && parts[0] == "@@" && parts[3] == "@@",
                "unsupported native patch hunk header"
            );
            let mut a = count(parts[1], '-');
            let mut b = count(parts[2], '+');
            assert!(a > 0 || b > 0, "empty native patch hunk");
            i += 1;
            while a > 0 || b > 0 {
                let body = lines.get(i).expect("truncated native patch hunk");
                if body.starts_with(' ') {
                    a = a.checked_sub(1).expect("native patch old hunk overflow");
                    b = b.checked_sub(1).expect("native patch new hunk overflow");
                } else if body.starts_with('-') {
                    a = a.checked_sub(1).expect("native patch old hunk overflow");
                } else if body.starts_with('+') {
                    b = b.checked_sub(1).expect("native patch new hunk overflow");
                } else {
                    assert_eq!(
                        *body, "\\ No newline at end of file",
                        "unsupported native patch hunk body"
                    );
                }
                i += 1;
            }
            if lines.get(i) == Some(&"\\ No newline at end of file") {
                i += 1;
            }
            hunks += 1;
            continue;
        } else if let Some(index) = line.strip_prefix("index ") {
            assert!(
                git_paths.is_some() && !active && old.is_none(),
                "unsupported native patch index placement"
            );
            let parts: Vec<_> = index.split_whitespace().collect();
            assert!(
                parts.len() == 1 || parts.len() == 2,
                "unsupported native patch index"
            );
            let (a, b) = parts[0]
                .split_once("..")
                .expect("native patch index hashes");
            assert!(
                !a.is_empty()
                    && !b.is_empty()
                    && a.bytes().chain(b.bytes()).all(|c| c.is_ascii_hexdigit()),
                "unsupported native patch index hashes"
            );
            assert!(
                parts.len() == 1 || ["100644", "100755"].contains(&parts[1]),
                "unsupported native patch index mode"
            );
        } else if line == "new file mode 100644"
            || line == "new file mode 100755"
            || line == "deleted file mode 100644"
            || line == "deleted file mode 100755"
        {
            assert!(
                git_paths.is_some() && !active && old.is_none(),
                "unsupported native patch mode placement"
            );
        } else if line == "-- "
            && mail
            && active
            && hunks > 0
            && old.is_none()
            && git_paths.is_none()
        {
            assert!(!footer, "duplicate native patch mail footer");
            footer = true;
        } else if footer && !line.is_empty() {
            assert!(
                !footer_version
                    && line.starts_with(|c: char| c.is_ascii_digit())
                    && line
                        .bytes()
                        .all(|c| c.is_ascii_alphanumeric() || b". ()-".contains(&c)),
                "unsupported native patch mail footer"
            );
            footer_version = true;
        } else if mail && files.is_empty() && !active && old.is_none() && git_paths.is_none() {
            // Git mail headers/message/diffstat are a supported envelope only
            // before the first section. Applicable file headers are always
            // parsed above, even here; context/binary/other diff formats refuse.
            let unquoted = line.trim_start_matches([' ', '\t', '>']);
            assert!(
                unquoted == "---"
                    || ![
                        "diff ",
                        "diff\t",
                        "---",
                        "+++",
                        "***",
                        "@@",
                        "Index",
                        "GIT binary",
                        "Binary"
                    ]
                    .iter()
                    .any(|prefix| unquoted.starts_with(prefix)),
                "unsupported native patch representation: {line}"
            );
        } else {
            assert!(
                line.is_empty(),
                "unsupported native patch representation: {line}"
            );
        }
        i += 1;
    }
    assert!(
        old.is_none() && git_paths.is_none() && active && hunks > 0 && !files.is_empty(),
        "incomplete native patch section"
    );
    files
}

pub fn script(
    patches: &[(String, String)],
    upstream_baseline: &str,
    patched_baseline: &str,
) -> String {
    assert!(!patches.is_empty());
    let mut files = BTreeSet::new();
    for (_, patch) in patches {
        files.extend(patch_paths(patch));
    }
    assert!(!files.is_empty());
    for baseline in [upstream_baseline, patched_baseline] {
        let lines: Vec<_> = baseline.lines().collect();
        assert_eq!(
            lines.len(),
            files.len() * 2,
            "baseline must cover every touched path"
        );
        for (pair, file) in lines.chunks_exact(2).zip(files.iter()) {
            assert_eq!(pair[0], format!("source:{file}"));
            assert!(
                pair[1] == "MISSING"
                    || (pair[1].len() == 64 && pair[1].bytes().all(|b| b.is_ascii_hexdigit()))
            );
        }
    }
    let mut script = String::from(
        "#!/bin/sh\nset -eu\nLC_ALL=C\nexport LC_ALL\n\
         d=\"$(dirname \"$0\")\"\n\
         fail() { echo \"pmetal: native patch integrity: $*; use a new owned native stage for these inputs\" >&2; exit 1; }\n\
         receipt=.pmetal-patch-series-v1\n\
         tmp=$(mktemp .pmetal-patch-candidate.XXXXXX)\n\
         baseline=$(mktemp .pmetal-patch-baseline.XXXXXX)\n\
         trap 'rm -f \"$tmp\" \"$baseline\"' 0\n\
         hash_file() {\n  test ! -L \"$1\" || fail 'symlink input'\n\
           if test -f \"$1\"; then\n    result=$(cmake -E sha256sum \"$1\") || fail 'cannot hash input'\n    printf '%s\\n' \"${result%% *}\"\n\
           elif test ! -e \"$1\"; then printf '%s\\n' MISSING;\n\
           else fail 'non-regular input'; fi\n}\n\
         emit_sources() {\n",
    );
    for file in &files {
        script.push_str(&format!(
            "  printf '%s\\n' 'source:{file}'\n  hash_file {}\n",
            quoted(file)
        ));
    }
    script.push_str("}\nverify_final() {\ncat > \"$baseline\" <<'PMETAL_PATCHED_BASELINE'\n");
    script.push_str(patched_baseline);
    assert!(patched_baseline.ends_with('\n'));
    script.push_str("PMETAL_PATCHED_BASELINE\nemit_sources > \"$tmp\"\ncmp -s \"$baseline\" \"$tmp\" || fail 'patched source differs from expected final content'\n}\nemit_receipt() {\n  printf '%s\\n' 'pmetal-native-patch-series-v1'\n");
    for (name, _) in patches {
        quoted(name);
        script.push_str(&format!(
            "  printf '%s\\n' 'patch:{name}'\n  hash_file \"$d/{name}\"\n"
        ));
    }
    script.push_str(
        "  emit_sources\n}\nemit_receipt > \"$tmp\"\n\
         if test -e \"$receipt\" || test -L \"$receipt\"; then\n\
           test ! -L \"$receipt\" && test -f \"$receipt\" || fail 'invalid receipt'\n\
           cmp -s \"$receipt\" \"$tmp\" || fail 'patch inputs/order or native source changed'\n\
           verify_final\n\
           echo 'pmetal: native patch series verified; no-op reentry' >&2\n\
           exit 0\n\
         fi\n\
         cat > \"$baseline\" <<'PMETAL_UPSTREAM_BASELINE'\n",
    );
    script.push_str(upstream_baseline);
    assert!(upstream_baseline.ends_with('\n'));
    script.push_str("PMETAL_UPSTREAM_BASELINE\nemit_sources > \"$tmp\"\n\
         cmp -s \"$baseline\" \"$tmp\" || fail 'unreceipted source is not the pinned pristine baseline'\n");
    for (name, _) in patches {
        // patch works on both Git clones and no-.git source archives, without
        // discovering an unrelated parent repository. Failure may leave partial
        // files but NEVER a receipt; exact upstream/final hashes prevent blessing
        // that tree. Content-keyed owned staging preserves the previous version.
        script.push_str(&format!(
            "patch -s -p1 -t -N -i \"$d/{name}\" || fail 'required patch {name} failed'\n"
        ));
    }
    script.push_str("verify_final\nemit_receipt > \"$tmp\"\nmv -f \"$tmp\" \"$receipt\"\necho 'pmetal: native patch series applied and content-receipted' >&2\n");
    script
}
