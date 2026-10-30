# Workflow-only regression: no native library or model is built/executed here.
require 'minitest/autorun'
require 'yaml'

class ValidationProfilesTest < Minitest::Test
  ROOT = File.expand_path('../workflows', __dir__)
  SAFE_TEST_COMMAND = 'cargo test -p pmetal-mlx-rs --test load_errors --test load_progress -- --test-threads=1'.freeze

  def workflow(name)
    YAML.safe_load(File.read(File.join(ROOT, "#{name}.yml")))
  end

  def triggers(workflow)
    # Ruby's YAML 1.1 parser recognizes unquoted `on` as true.
    workflow.fetch('on') { workflow.fetch(true) }.keys.sort
  end

  def commands(workflow)
    workflow.fetch('jobs').values.flat_map { |job| job.fetch('steps', []) }
      .map { |step| step['run'] }.compact
  end

  def test_ordinary_validation_compiles_everything_but_executes_only_audited_io_targets
    ordinary = workflow('validate')
    assert_equal %w[pull_request push], triggers(ordinary)
    assert_equal ['checks'], ordinary.fetch('jobs').keys
    runs = commands(ordinary)
    assert_includes runs, 'cargo fmt --all -- --check'
    assert_includes runs, 'cargo clippy --workspace -- -D warnings'
    assert_includes runs, 'cargo test --all --no-run'
    setup = ordinary.fetch('jobs').fetch('checks').fetch('steps').find { |step| step.fetch('uses', '').start_with?('actions-rust-lang/setup-rust-toolchain@') }
    assert_equal '', setup.fetch('with').fetch('rustflags')
    assert_equal [
      'sudo xcodes select 16.4',
      'cargo --version',
      'ruby .github/tests/validation_profiles_test.rb',
      'cargo fmt --all -- --check',
      'cargo clippy --workspace -- -D warnings',
      'cargo clippy -p pmetal-mlx-rs --test load_errors --test load_progress -- -D warnings',
      'cargo test --all --no-run',
      SAFE_TEST_COMMAND
    ], runs
    refute ordinary.fetch('jobs').values.any? { |job| job.key?('uses') }
  end

  def test_full_matrix_and_native_retirement_are_manual_only_and_preserved
    manual = workflow('model-validation')
    assert_equal %w[workflow_call workflow_dispatch], triggers(manual)
    job = manual.fetch('jobs').fetch('tests')
    assert_equal "github.event_name == 'workflow_dispatch'", job.fetch('if')
    assert_equal ["1.85.0", 'stable'], job.fetch('strategy').fetch('matrix').fetch('rust')
    assert_equal ['macos-15'], job.fetch('strategy').fetch('matrix').fetch('runner')
    assert_equal 1, job.fetch('strategy').fetch('max-parallel')
    assert_equal 180, job.fetch('timeout-minutes')
    assert_equal false, manual.fetch('concurrency').fetch('cancel-in-progress')
    assert_equal 'model-validation-${{ github.run_id }}-${{ github.run_attempt }}', manual.fetch('concurrency').fetch('group')
    runs = commands(manual)
    assert_equal 1, runs.count { |run| run.include?('cargo test --all -- --test-threads=1') }
    retirement = job.fetch('steps').find { |step| step.fetch('name', '') == 'Focused native CPU and Metal owner retirement' }
    assert_equal "matrix.rust == 'stable'", retirement.fetch('if')
    assert_includes retirement.fetch('run'), 'bash mlx-sys/tests/validate_native_retirement.sh'
    assert job.fetch('steps').any? { |step| step.fetch('uses', '').start_with?('actions/upload-artifact@') && step['if'] == 'always()' }
  end

  def test_registered_dispatcher_isolates_validation_from_publication
    prebuilt = workflow('prebuilt')
    on = prebuilt.fetch('on') { prebuilt.fetch(true) }
    assert_equal ['main'], on.fetch('push').fetch('branches')
    input = on.fetch('workflow_dispatch').fetch('inputs').fetch('validation_only')
    assert_equal 'boolean', input.fetch('type')
    assert_equal false, input.fetch('default')
    call = prebuilt.fetch('jobs').fetch('model-validation')
    assert_equal "github.event_name == 'workflow_dispatch' && inputs.validation_only == true", call.fetch('if')
    assert_equal './.github/workflows/model-validation.yml', call.fetch('uses')
    assert_equal "github.event_name != 'workflow_dispatch' || inputs.validation_only != true", prebuilt.fetch('jobs').fetch('build').fetch('if')
    Dir.glob(File.join(ROOT, '*.yml')).each do |path|
      next if File.basename(path) == 'prebuilt.yml'
      document = YAML.safe_load(File.read(path))
      refute document.fetch('jobs').values.any? { |job| job['uses'] == './.github/workflows/model-validation.yml' }, path
    end
  end
end
