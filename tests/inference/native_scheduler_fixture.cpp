#include "recurrent_pipeline.h"
#include "recoverable_engine.h"
#include <crit/json.h>
#include <filesystem>
#include <fstream>
#include <thread>
namespace cc=crit::core;
namespace cp=crit::examples::pipeline;
namespace ca=crit::examples::attention;
namespace fs=std::filesystem;
struct Observations final : cp::Hooks {
  fs::path root;
  explicit Observations(fs::path p):root(std::move(p)) {}
  void computed(cc::StepKey key) override {
    std::ofstream(root/("computed-"+std::to_string(key.request.request))).put('1');
    // Bounded synthetic arithmetic latency lets ordinary HTTP clients overlap.
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
};
template<class Base> struct ObservedModel final : Base {
  fs::path root;
  template<class... Args> ObservedModel(fs::path p,Args&&... args):Base(std::forward<Args>(args)...),root(std::move(p)) {}
  cc::Admission admission(const cc::RequestView& view) override {
    std::ofstream(root/("queued-"+std::to_string(view.key.request))).put('1');
    if(!fs::exists(root/"release")) return cc::Admission::deferred;
    return Base::admission(view);
  }
  std::unique_ptr<cc::Retirement> retire(cc::RequestKey key,cc::ConcurrentSession& session,cc::Finish finish) override {
    auto receipt=Base::retire(key,session,finish);
    std::ofstream(root/("retired-"+std::to_string(key.request)))<<int(finish);
    return receipt;
  }
};
cc::NativeEngine create(std::string_view config) {
  const auto doc=cc::json::Document::parse(config,"scheduler fixture");
  const auto* directory=doc.root().find("control_directory");
  if(!directory) throw std::invalid_argument("control_directory required");
  const fs::path root=std::string(directory->string());
  const auto* option=doc.root().find("attention");
  const bool attention=option && option->boolean();
  fs::create_directories(root);
  cc::Profile p{64,2,8*1024*1024,1024,2,2}; p.max_batch_tokens=4; p.max_inflight_batches=6; p.memory_preemption=true;
  p.admission_lookahead=2; // Let the fixture observe both queued requests before releasing admission.
  if(!attention) { p.max_request_steps=3; p.max_step_output_tokens=2; }
  cc::NativeEngine native; native.concurrent_requests=true;
  native.engine=std::make_unique<cc::Engine>(p,std::make_shared<ca::Backend>(),[root,attention](const auto&,cc::Services services)->std::unique_ptr<cc::Model> {
    if(attention) {
      auto stats=std::make_shared<ca::Stats>(); stats->start=true;
      return std::make_unique<ObservedModel<ca::RecoverableModel>>(root,stats,services.resources,ca::recovery_config());
    }
    return std::make_unique<ObservedModel<cp::Model>>(root,services.resources,3,std::make_shared<Observations>(root),true);
  });
  native.result=[root](const cc::Result& result) {
    const auto text="{\"id\":"+std::to_string(result.request_id)+",\"preemptions\":"+std::to_string(result.timing.preemptions)+
        ",\"output\":"+std::to_string(result.tokens.size())+"}";
    std::ofstream(root/("result-"+std::to_string(result.request_id)))<<text;
    return text;
  };
  return native;
}
CRIT_EXPORT_ENGINE(create)
