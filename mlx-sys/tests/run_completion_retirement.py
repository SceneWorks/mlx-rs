#!/usr/bin/env python3
"""CPU-only actual callback/range/receipt/finalization tests and compiling mutation reds."""
import argparse, os, resource, subprocess
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--work-dir',type=Path,required=True)
p.add_argument('--native-source',type=Path)
a=p.parse_args();resource.setrlimit(resource.RLIMIT_CORE,(0,0))
root=Path(__file__).resolve().parents[1]
paths=['mlx/backend/metal/completion_retirement.h','mlx/backend/metal/commit_completion.h','mlx/retirement.h','mlx/backend/common/execution_outcome.h','mlx/backend/common/retirement_capture.h']
patch=(root/'patches/completion-retirement.patch').read_text() if not a.native_source else None
headers={}
for path in paths:
 if a.native_source:headers[path]=(a.native_source/path).read_text()
 else:
  section=patch.split('+++ b/'+path+'\n',1)[1].split('diff --git ',1)[0]
  headers[path]=''.join(line[1:]+'\n' for line in section.splitlines() if line.startswith('+'))
def variant(path,old,new,count=-1):
 result=dict(headers);result[path]=result[path].replace(old,new,count);return result
h,c,r,e,pipeline=paths
mutations={
 'green':headers,
 'retain-body-owners':variant(h,'body_ = nullptr;','(void)body_;'),
 'publish-before-drop':variant(h,'body_ = nullptr; // owner destruction BEFORE completion publication\n        record_->state_->finish(record_->ticket_);','record_->state_->finish(record_->ticket_);\n        body_ = nullptr;'),
 'skip-range-wait':variant(h,'ready_.wait(lock, [&] { return complete_locked(through); });','(void)through;'),
 'latest-only-wait':variant(h,'if (entry.status != Status::Complete) return false;','if (ticket < through) continue;\n      if (entry.status != Status::Complete) return false;'),
 'lose-callback-error':variant(h,'entry.error = std::move(error);','(void)error;'),
 'consume-cached-receipt':variant(r,'if (state_->error) std::rethrow_exception(state_->error);','if (auto error = std::exchange(state_->error, {})) std::rethrow_exception(error);'),
 'ack-on-unobserved-drop':variant(r,'    std::exception_ptr error;','    ~State() { acknowledge(); }\n    std::exception_ptr error;',1),
 'ack-beyond-captured-range':variant(h,'auto end = entries_.upper_bound(through);','auto end = entries_.end();'),
 'ack-unretired':variant(h,'if (!complete_locked(through)) return false;','(void)through;'),
 'ack-reserved':variant(h,'if (entry.status != Status::Complete) return false;','if (entry.status == Status::Pending) return false;'),
 'execution-report-clears-callback':variant(h,'entry.kind == Kind::Execution && entry.key == key','entry.key == key'),
 'erase-pending-execution-record':variant(h,'if (entry.status == Status::Complete) { it = entries_.erase(it); continue; }','{ it = entries_.erase(it); continue; }'),
 'lose-callback-upgrade':variant(h,'entry.kind == Kind::Execution && kind == Kind::Callback','entry.kind == Kind::Execution && kind == Kind::Execution'),
 'skip-commit-finalization':variant(c,'  finalize();','  (void)finalize;'),
 'skip-accounting-on-body-error':variant(c,'try { body(); } catch (...) { on_error(std::current_exception()); }','try { body(); } catch (...) { on_error(std::current_exception()); finalize(); return; }'),
 'lose-commit-exception':variant(c,'on_error(std::current_exception());','(void)on_error;'),
 'double-commit-finalization':variant(c,'  finalize();','  finalize(); finalize();'),
 'capture-before-producer-submit':variant(pipeline,'  submit();\n  return capture();','  auto receipt = capture();\n  submit();\n  return receipt;'),
 'ignore-failed-submit':variant(pipeline,'  submit();','  try { submit(); } catch (...) {}'),
 'ack-after-failed-submit':variant(pipeline,'  submit();','  try { submit(); } catch (...) { auto failed = capture(); try { failed.wait(); } catch (...) {} throw; }'),
}
a.work_dir.mkdir(parents=True,exist_ok=False)
for name,sources in mutations.items():
 assert name=='green' or sources!=headers,'invalid unchanged mutation '+name
 directory=a.work_dir/name
 for path,text in sources.items():
  dest=directory/path;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(text)
 binary=directory/'ownership-test'
 subprocess.run(['nice','-n','10',os.environ.get('CXX','clang++'),'-std=c++17','-O0','-g0','-pthread','-Wall','-Wextra','-Werror','-I'+str(directory),str(root/'tests/completion_retirement_test.cpp'),'-o',str(binary)],check=True)
 try:
  result=subprocess.run(['nice','-n','10',str(binary)],capture_output=True,text=True,timeout=12);code,output=result.returncode,result.stdout+result.stderr
 except subprocess.TimeoutExpired:code,output=124,'ownership/range/finalization timeout\n'
 (directory/'result.txt').write_text(f'exit={code}\n{output}')
 if name=='green':assert code==0,output;print(output.strip(),flush=True)
 else:assert code!=0,'MUTATION SURVIVED '+name;print(f'compiling mutation RED: {name}: exit={code}: {output.strip()}',flush=True)
print(f'PASS: production receipt/ownership headers green; {len(mutations)-1} valid compiling mutation reds',flush=True)
