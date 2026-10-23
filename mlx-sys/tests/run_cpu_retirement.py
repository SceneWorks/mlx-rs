#!/usr/bin/env python3
"""CPU production queue/array/custody/range tests; COMPLETE patched native source required. No native link/GPU."""
import argparse,resource,subprocess
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--native-source',type=Path,required=True);p.add_argument('--work-dir',type=Path,required=True);a=p.parse_args();resource.setrlimit(resource.RLIMIT_CORE,(0,0));a.work_dir.mkdir(parents=True,exist_ok=False)
root=Path(__file__).resolve().parents[1]
paths=['mlx/scheduler.h','mlx/backend/common/execution_outcome.h','mlx/backend/cpu/deferred_outcome.h','mlx/load_completion.h','mlx/backend/metal/completion_retirement.h','mlx/retirement.h']
headers={path:(a.native_source/path).read_text()for path in paths}
def variant(path,old,new,count=-1):
 result=dict(headers);result[path]=result[path].replace(old,new,count);return result
s,o,c,l,t,r=paths
mutations={
 'green':headers,
 'parallel-not-FIFO':variant(s,'      task(); // production payload catches callback errors and destroys its body','      std::thread(std::move(task)).detach();'),
 'skip-checkpoint':variant(s,'return retired_tasks >= through;','(void)through; return true;'),
 'checkpoint-before-work':variant(s,'      task(); // production payload catches callback errors and destroys its body\n      task = nullptr;\n      { std::lock_guard lock(mtx); ++retired_tasks; }\n      retirement_cv.notify_all();','      { std::lock_guard lock(mtx); ++retired_tasks; }\n      retirement_cv.notify_all();\n      task();\n      task = nullptr;'),
 'wrong-captured-watermark':variant(s,'through = retirement->snapshot();','through = 0;'),
 'cancel-published-work':variant(s,'q.emplace([callback] { callback->run(0); });','q.emplace([callback] { callback->cancel_unpublished(); });'),
 'successful-failed-array':variant(o,'promise_.set_exception(error_);','promise_.set_value();'),
 'lose-primitive-failure':variant(o,'record_error(std::current_exception());','(void)std::current_exception();'),
 'cancel-already-queued-work':variant(o,'    try { task(); }','    if ([&] { std::lock_guard lock(mutex_); return error_; }()) return nullptr;\n    try { task(); }'),
 'drop-custody-on-defer':variant(c,'head_ = std::move(node);','node.reset();'),
 'retain-custody-cycle':variant(c,'head_ = std::move(node->next);','head_ = std::move(node->next); node->next = node;'),
 'lose-late-array-error':variant(l,'if (!error_) error_ = std::move(error);','(void)error;',1),
 'failed-setup-waits-for-producer':variant(l,'if (failed_setup_) std::rethrow_exception(error_);','(void)failed_setup_;'),
 'known-failed-input-accepted':variant(l,'if (error) std::rethrow_exception(error);','(void)error;'),
}
for name,sources in mutations.items():
 assert name=='green' or sources!=headers,'invalid unchanged mutation '+name
 directory=a.work_dir/name
 for path,text in sources.items():
  target=directory/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(text)
 binary=directory/'cpu-ownership-test';subprocess.run(['nice','-n','10','clang++','-std=c++20','-O0','-g0','-pthread','-I'+str(directory),'-I'+str(a.native_source),str(root/'tests/cpu_retirement_test.cpp'),'-o',str(binary)],check=True)
 try:
  result=subprocess.run(['nice','-n','10',str(binary)],capture_output=True,text=True,timeout=10);code,output=result.returncode,result.stdout+result.stderr
 except subprocess.TimeoutExpired:code,output=124,'CPU owner/range timeout\n'
 (directory/'result.txt').write_text(f'exit={code}\n{output}')
 if name=='green':assert code==0,output;print(output.strip(),flush=True)
 else:assert code!=0,'MUTATION SURVIVED '+name;print(f'compiling CPU mutation RED: {name}: exit={code}: {output.strip()}',flush=True)
print(f'PASS: production CPU queue/array/range green + {len(mutations)-1} valid compiling mutation reds',flush=True)
