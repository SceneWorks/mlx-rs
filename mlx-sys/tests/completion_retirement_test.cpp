// CPU-only tests of ACTUAL production callback journal, receipts and finalizer.
#include "mlx/backend/metal/completion_retirement.h"
#include "mlx/backend/metal/commit_completion.h"
#include "mlx/backend/common/execution_outcome.h"
#include "mlx/backend/common/retirement_capture.h"
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <future>
#include <iostream>
#include <stdexcept>
#include <thread>
using mlx::core::metal::CompletionRetirement;
using mlx::core::RetirementFence;
using namespace std::chrono_literals;
static void require(bool ok,const char* message) { if(!ok){std::cerr<<message<<'\n';std::abort();} }
template<typename F> static void error_is(F&& f,const char* text) {
  try {f();require(false,"failed receipt/array became successful");}
  catch(const std::runtime_error& e){require(std::string(e.what())==text,"wrong cached failure");}
}
struct Gate {
 std::promise<void> entered,open;
 std::shared_future<void> opened=open.get_future().share();
 void block(){entered.set_value();opened.wait();}
 void await_entry(){require(entered.get_future().wait_for(2s)==std::future_status::ready,"callback/destructor never entered");}
};
struct DropToken {Gate* gate;std::atomic<int>* drops;~DropToken(){if(gate)gate->block();++*drops;}};
static void fail(const std::shared_ptr<CompletionRetirement>& state,const char* text){
 auto cb=state->track<int>([text](int){throw std::runtime_error(text);});cb->run(0);
}
static void delayed_body_and_destructor(){
 auto state=std::make_shared<CompletionRetirement>();Gate body,destructor;std::atomic<int> drops{0};
 auto owner=std::make_shared<DropToken>();owner->gate=&destructor;owner->drops=&drops;std::weak_ptr<DropToken> weak=owner;
 auto payload=state->track<int>([owner=std::move(owner),&body](int){body.block();});
 std::function<void()> metal_copy=[payload]{payload->run(0);};auto another=metal_copy;
 auto callback=std::async(std::launch::async,metal_copy);body.await_entry();
 auto capture=std::async(std::launch::async,[state]{return state->fence();});
 require(capture.wait_for(50ms)==std::future_status::timeout,"fence returned before callback body completed");
 body.open.set_value();destructor.await_entry();
 require(capture.wait_for(50ms)==std::future_status::timeout,"fence returned before captured owner destructor completed");
 destructor.open.set_value();auto fence=capture.get();fence.wait();callback.get();
 require(weak.expired()&&drops==1,"retained outer handler copies retain owners");another();fence.wait();require(drops==1,"duplicate invocation released twice");
}
static void out_of_order_and_range(){
 auto state=std::make_shared<CompletionRetirement>();auto first=state->track<int>([](int){});auto second=state->track<int>([](int){});
 auto through=state->snapshot();auto later=state->track<int>([](int){});second->run(0);
 auto waiter=std::async(std::launch::async,[state,through]{state->wait(through);});
 require(waiter.wait_for(50ms)==std::future_status::timeout,"latest callback is not a fence for earlier callbacks");
 first->run(0);require(waiter.wait_for(2s)==std::future_status::ready,"range incorrectly waits for later callbacks");waiter.get();
 require(state->acknowledge(through),"completed range not acknowledged");require(!state->acknowledge(state->snapshot()),"old range acknowledged later unretired callback");
 later->run(0);state->fence().wait();
}
static void errors_release_before_reporting_and_cache(){
 auto state=std::make_shared<CompletionRetirement>();Gate destructor;std::atomic<int> drops{0};
 auto owner=std::make_shared<DropToken>();owner->gate=&destructor;owner->drops=&drops;
 auto cb=state->track<int>([owner=std::move(owner)](int){throw std::runtime_error("first async error");});
 auto run=std::async(std::launch::async,[cb]{cb->run(0);});destructor.await_entry();
 auto capture=std::async(std::launch::async,[state]{return state->fence();});
 require(capture.wait_for(50ms)==std::future_status::timeout,"error receipt returned before owner release");
 destructor.open.set_value();auto receipt=capture.get();run.get();require(drops==1,"exception leaked owner");
 error_is([&]{receipt.wait();},"first async error");error_is([&]{receipt.wait();},"first async error");
 state->fence().wait(); // same stream, new acknowledged range recovers
 error_is([&]{receipt.wait();},"first async error");
}
static void aliases_and_multiple_obligations(){
 auto state=std::make_shared<CompletionRetirement>();std::atomic<int> drops{0};auto alias=std::make_shared<DropToken>();alias->gate=nullptr;alias->drops=&drops;
 std::weak_ptr<DropToken> weak=alias;auto first=state->track<int>([owner=alias](int){});auto second=state->track<int>([owner=alias](int){});
 first->run(0);second->run(0);state->fence().wait();require(!weak.expired()&&drops==0&&alias.use_count()==1,"retirement freed external alias or retained native aliases");
 alias.reset();require(weak.expired()&&drops==1,"external alias did not release");
}
static void commit_failure_finalizes_exactly_once(){
 std::promise<void> event;auto signaled=event.get_future();int accounting=0,finalized=0;std::exception_ptr error;
 mlx::core::metal::run_commit_completion([]{throw std::bad_alloc();},[&]{++accounting;},
  [&]()noexcept{++finalized;event.set_value();},[&](std::exception_ptr e)noexcept{if(!error)error=std::move(e);});
 require(signaled.wait_for(50ms)==std::future_status::ready,"throwing callback skipped finalization");
 require(accounting==1&&finalized==1,"throwing callback skipped/duplicated accounting/finalization");require(error!=nullptr,"callback exception lost");
 try{std::rethrow_exception(error);}catch(const std::bad_alloc&){return;}require(false,"wrong callback exception");
}
static void commit_finalize_then_owner_retirement(){
 auto state=std::make_shared<CompletionRetirement>();auto record=state->prepare();Gate destructor;std::atomic<int> drops{0};
 auto owner=std::make_shared<DropToken>();owner->gate=&destructor;owner->drops=&drops;std::promise<void> event;auto signaled=event.get_future();int accounting=0;
 auto cb=state->track<int>([owner=std::move(owner),record,&event,&accounting](int){
  mlx::core::metal::run_commit_completion([]{throw std::runtime_error("commit body failed");},[&]{++accounting;throw std::bad_alloc();},
   [&]()noexcept{event.set_value();},[&](std::exception_ptr e)noexcept{record->error(std::move(e));});},record);
 auto run=std::async(std::launch::async,[cb]{cb->run(0);});require(signaled.wait_for(2s)==std::future_status::ready,"commit failure stranded event");destructor.await_entry();
 auto capture=std::async(std::launch::async,[state]{return state->fence();});require(capture.wait_for(50ms)==std::future_status::timeout,"finalization bypassed owner destructor");
 destructor.open.set_value();auto receipt=capture.get();run.get();require(accounting==1&&drops==1,"commit failure leaked/duplicated obligations");error_is([&]{receipt.wait();},"commit body failed");
}
static void dropping_unobserved_receipt_does_not_ack(){
 auto state=std::make_shared<CompletionRetirement>();fail(state,"unobserved");{auto dropped=state->fence();}
 auto observed=state->fence();error_is([&]{observed.wait();},"unobserved");state->fence().wait();error_is([&]{observed.wait();},"unobserved");
}
static void overlapping_and_out_of_order_receipts(){
 auto state=std::make_shared<CompletionRetirement>();fail(state,"older");auto old=state->fence();fail(state,"newer");auto overlap=state->fence();
 error_is([&]{old.wait();},"older");auto remaining=state->fence();error_is([&]{remaining.wait();},"newer");
 error_is([&]{overlap.wait();},"older");fail(state,"uncaptured");error_is([&]{old.wait();},"older");error_is([&]{overlap.wait();},"older");
 auto last=state->fence();error_is([&]{last.wait();},"uncaptured");state->fence().wait();
 fail(state,"out of order");auto a=state->fence();auto b=state->fence();error_is([&]{b.wait();},"out of order");error_is([&]{a.wait();},"out of order");state->fence().wait();
}
static void reserved_and_unretired_cannot_be_acknowledged(){
 auto state=std::make_shared<CompletionRetirement>();auto reserved=state->prepare();auto through=state->snapshot();require(!state->acknowledge(through),"reserved obligation acknowledged before publication/cancellation");
 reserved.reset();state->wait(through);require(state->acknowledge(through),"cancelled reserved record stuck");
 auto cb=state->track<int>([](int){});through=state->snapshot();require(!state->acknowledge(through),"pending callback acknowledged before retirement");cb->cancel_unpublished();state->fence().wait();
}
static void execution_ack_does_not_clear_callback_failure(){
 auto state=std::make_shared<CompletionRetirement>();auto key=std::make_shared<std::string>("gpu failure");auto record=state->prepare();
 auto execution=std::make_shared<mlx::core::detail::ExecutionOutcome>();auto future=execution->future();
 auto cb=state->track<int>([record,key,state,execution](int){auto e=std::make_exception_ptr(std::runtime_error("array execution failed"));execution->record_error(e);execution->finish();record->error(e,CompletionRetirement::Kind::Execution,key);},record);cb->run(0);
 state->acknowledge_execution(key);state->fence().wait();for(int i=0;i<2;++i)error_is([&]{future.get();},"array execution failed");
 auto structural=state->prepare();auto bad=state->track<int>([structural,state,key](int){
  structural->error(std::make_exception_ptr(std::runtime_error("execution")),CompletionRetirement::Kind::Execution,key);
  state->acknowledge_execution(key); // report while callback still pending
  structural->error(std::make_exception_ptr(std::runtime_error("callback")),CompletionRetirement::Kind::Callback,key);
 },structural);bad->run(0);state->acknowledge_execution(key);auto failed=state->fence();error_is([&]{failed.wait();},"callback");state->fence().wait();error_is([&]{failed.wait();},"callback");
 auto healthy=std::make_shared<mlx::core::detail::ExecutionOutcome>();healthy->finish();healthy->future().get();
}
static void capture_includes_submission_created_callbacks(){
 auto state=std::make_shared<CompletionRetirement>();Gate destructor;std::atomic<int>drops{0};
 auto owner=std::make_shared<DropToken>();owner->gate=&destructor;owner->drops=&drops;
 using Callback=CompletionRetirement::Callback<int>;
 std::promise<std::shared_ptr<Callback>> published;auto get=published.get_future();
 auto capture=std::async(std::launch::async,[state,owner=std::move(owner),&published]()mutable{
  return mlx::core::detail::capture_retirement_range([&]{published.set_value(state->track<int>([owner=std::move(owner)](int){}));},[&]{return state->fence();});
 });
 auto cb=get.get();auto run=std::async(std::launch::async,[cb]{cb->run(0);});destructor.await_entry();
 bool blocked=capture.wait_for(50ms)==std::future_status::timeout;destructor.open.set_value();auto receipt=capture.get();run.get();
 require(blocked,"capture snapshot preceded producer-created callback");receipt.wait();require(drops==1,"submission owner not retired");
}
static void failed_submission_never_captures_or_acknowledges(){
 auto state=std::make_shared<CompletionRetirement>();fail(state,"unreported before submit");
 try{mlx::core::detail::capture_retirement_range([]{throw std::bad_alloc();},[&]{return state->fence();});require(false,"failed submit produced completed receipt");}
 catch(const std::bad_alloc&){}
 auto receipt=state->fence();error_is([&]{receipt.wait();},"unreported before submit");state->fence().wait();
}
int main(){
 delayed_body_and_destructor();out_of_order_and_range();errors_release_before_reporting_and_cache();aliases_and_multiple_obligations();
 commit_failure_finalizes_exactly_once();commit_finalize_then_owner_retirement();dropping_unobserved_receipt_does_not_ack();overlapping_and_out_of_order_receipts();
 reserved_and_unretired_cannot_be_acknowledged();execution_ack_does_not_clear_callback_failure();
 capture_includes_submission_created_callbacks();failed_submission_never_captures_or_acknowledges();
 std::cout<<"12 production callback/range/receipt/finalization/capture tests passed\n";
}
