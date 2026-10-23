// CPU-only tests of ACTUAL StreamThread, execution DAGs, custody and receipts.
#include "mlx/scheduler.h"
#include "mlx/backend/cpu/task_outcome.h"
#include "mlx/backend/cpu/deferred_outcome.h"
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
using mlx::core::scheduler::StreamThread;
using namespace std::chrono_literals;
static void require(bool ok,const char* text){if(!ok){std::cerr<<text<<'\n';std::abort();}}
template<typename F>static void error_is(F&& f,const char* text){try{f();require(false,"failed result became successful");}catch(const std::runtime_error& e){require(std::string(e.what())==text,"wrong persistent error");}}
struct Gate{std::promise<void> entered,release;std::shared_future<void>opened=release.get_future().share();void block(){entered.set_value();opened.wait();}void await_entry(){require(entered.get_future().wait_for(2s)==std::future_status::ready,"prior queued owner destructor did not run");}};
struct Owner{Gate*gate;std::atomic<int>*drops;~Owner(){if(gate)gate->block();++*drops;}};
static std::future<void> sentinel(StreamThread& q){auto p=std::make_shared<std::promise<void>>();auto f=p->get_future();q.enqueue([p=std::move(p)]{p->set_value();});return f;}
static void prior_destructor_before_sentinel(){
 StreamThread q;Gate d;std::atomic<int>drops{0};auto owner=std::make_shared<Owner>();owner->gate=&d;owner->drops=&drops;std::weak_ptr<Owner>weak=owner;
 q.enqueue([owner=std::move(owner)]{});d.await_entry();auto done=sentinel(q);require(done.wait_for(50ms)==std::future_status::timeout,"FIFO sentinel completed before earlier owner destructor");d.release.set_value();done.get();require(weak.expired()&&drops==1,"earlier queue owners remain");q.retirement_fence().wait();
}
static void throwing_callback_drops_progresses_and_receipt_recovers(){
 StreamThread q;Gate d;std::atomic<int>drops{0};auto owner=std::make_shared<Owner>();owner->gate=&d;owner->drops=&drops;
 q.enqueue([owner=std::move(owner)]{throw std::runtime_error("callback failure");});d.await_entry();auto capture=std::async(std::launch::async,[&q]{return q.retirement_fence();});
 require(capture.wait_for(50ms)==std::future_status::timeout,"failed queue callback bypassed owner destruction");d.release.set_value();auto failed=capture.get();
 error_is([&]{failed.wait();},"callback failure");error_is([&]{failed.wait();},"callback failure");int value=0;q.enqueue([&]{value=7;});q.retirement_fence().wait();require(value==7&&drops==1,"same queue did not recover after observed range");error_is([&]{failed.wait();},"callback failure");
}
static void array_load_failure_does_not_poison_retirement(){
 StreamThread q;std::promise<void>io;auto read=io.get_future().share();io.set_exception(std::make_exception_ptr(std::runtime_error("bad read")));
 auto bad=std::make_shared<mlx::core::LoadCompletion>();bad->add(read);q.enqueue([read]{read.wait();});sentinel(q).get();error_is([&]{bad->wait();},"bad read");
 int value=0;q.enqueue([&]{value=9;});q.retirement_fence().wait();require(value==9,"independent request body failed");error_is([&]{bad->wait();},"bad read");
 mlx::core::LoadCompletion child;child.depend_on(bad);error_is([&]{child.wait();},"bad read");q.retirement_fence().wait();
}
static void execution_failure_follows_arrays_not_stream(){
 StreamThread q;auto outcome=std::make_shared<mlx::core::cpu::TaskOutcome>();auto bad=std::make_shared<mlx::core::LoadCompletion>();bad->add(outcome->future());
 q.enqueue([outcome]{auto body=[]{throw std::runtime_error("primitive failed");};outcome->run(body);});q.enqueue([outcome]{outcome->finish();});sentinel(q).get();
 mlx::core::LoadCompletion child;child.depend_on(bad);for(int i=0;i<3;++i){error_is([&]{bad->wait();},"primitive failed");error_is([&]{child.wait();},"primitive failed");}
 q.retirement_fence().wait();int healthy=0;q.enqueue([&]{healthy=5;});q.retirement_fence().wait();require(healthy==5,"ordinary primitive error poisoned later retirement");
 std::promise<void> ready;mlx::core::LoadCompletion late;late.add(ready.get_future().share());ready.set_value();late.wait();late.record_setup_error(std::make_exception_ptr(std::bad_alloc()));
 for(int i=0;i<2;++i){try{late.wait();require(false,"late setup failure bypassed ready outcome");}catch(const std::bad_alloc&){}}
}
static void queued_work_and_destruction_before_failed_outcome(){
 StreamThread q;auto outcome=std::make_shared<mlx::core::cpu::TaskOutcome>();outcome->record_error(std::make_exception_ptr(std::runtime_error("host failed")));Gate d;std::atomic<int>drops{0},runs{0};
 auto owner=std::make_shared<Owner>();owner->gate=&d;owner->drops=&drops;q.enqueue([outcome,owner=std::move(owner),&runs]{auto body=[&runs]{++runs;};outcome->run(body);});q.enqueue([outcome]{outcome->finish();});d.await_entry();
 require(outcome->future().wait_for(50ms)==std::future_status::timeout,"outcome ready before prior captures destroyed");d.release.set_value();q.retirement_fence().wait();require(runs==1&&drops==1,"host failure cancelled already queued work");error_is([&]{outcome->future().get();},"host failed");
}
static void nonallocating_checkpoint_joins_destruction(){
 StreamThread q;Gate d;std::atomic<int>drops{0};auto owner=std::make_shared<Owner>();owner->gate=&d;owner->drops=&drops;q.enqueue([owner=std::move(owner)]{});d.await_entry();
 auto wait=std::async(std::launch::async,[&]{q.wait_for_queued_tasks();});require(wait.wait_for(50ms)==std::future_status::timeout,"checkpoint returned before queued owner destruction");d.release.set_value();wait.get();require(drops==1,"checkpoint lost owner");
}
static void setup_custody_and_failed_parent_readiness(){
 using Held=std::shared_ptr<Owner>;using Node=mlx::core::cpu::DeferredOutcome<Held>;mlx::core::cpu::DeferredOutcomes<Held>custody;Gate d;std::atomic<int>drops{0};
 auto owner=std::make_shared<Owner>();owner->gate=nullptr;owner->drops=&drops;std::weak_ptr<Owner>weak=owner;auto node=std::make_shared<Node>();node->retain(owner,std::vector<Held>{owner},std::vector<Held>{owner});node->record_error(std::make_exception_ptr(std::bad_alloc()));auto future=node->future();node->finish();node->finish();owner.reset();custody.defer(std::move(node));require(!weak.expired()&&drops==0,"failed readiness dropped raw-work custody");
 std::promise<void>producer;auto parent=std::make_shared<mlx::core::LoadCompletion>();parent->add(producer.get_future().share());mlx::core::LoadCompletion bad;bad.depend_on(parent);bad.add(future);bad.record_setup_error(std::make_exception_ptr(std::bad_alloc()));
 auto wait=std::async(std::launch::async,[&]{bad.wait();});bool ready=wait.wait_for(50ms)==std::future_status::ready;producer.set_value();try{wait.get();require(false,"failed setup succeeded");}catch(const std::bad_alloc&){}require(ready,"failed setup waited on unsubmitted producer");require(!weak.expired()&&drops==0,"logical failure released physical owner");try{bad.check_known_error();require(false,"known failed input accepted");}catch(const std::bad_alloc&){}
 auto alias=weak.lock();alias->gate=&d;alias.reset();auto release=std::async(std::launch::async,[&]{custody.release();});d.await_entry();require(release.wait_for(50ms)==std::future_status::timeout,"custody release returned before destructor");d.release.set_value();release.get();require(drops==1&&weak.expired(),"custody retained owner");
}
static void unobserved_and_overlapping_cpu_receipts(){
 StreamThread q;q.enqueue([]{throw std::runtime_error("old");});{auto dropped=q.retirement_fence();}auto old=q.retirement_fence();q.enqueue([]{throw std::runtime_error("new");});auto overlap=q.retirement_fence();error_is([&]{old.wait();},"old");auto newer=q.retirement_fence();error_is([&]{newer.wait();},"new");error_is([&]{overlap.wait();},"old");
 q.enqueue([]{throw std::runtime_error("outside");});error_is([&]{old.wait();},"old");auto outside=q.retirement_fence();error_is([&]{outside.wait();},"outside");q.retirement_fence().wait();
}
int main(){prior_destructor_before_sentinel();throwing_callback_drops_progresses_and_receipt_recovers();array_load_failure_does_not_poison_retirement();execution_failure_follows_arrays_not_stream();queued_work_and_destruction_before_failed_outcome();nonallocating_checkpoint_joins_destruction();setup_custody_and_failed_parent_readiness();unobserved_and_overlapping_cpu_receipts();std::cout<<"8 production CPU queue/array/custody/range-recovery tests passed\n";}
