#include <private/GigaLearnCPP/Util/Models.h>
#include <torch/serialize.h>
#include <cassert>
#include <chrono>
#include <sstream>
using GGL::Muon;
using GGL::MuonOptions;
using Params = std::vector<torch::Tensor>;
Params initial() {
    return {torch::randn({4,3}).set_requires_grad(true), torch::randn({4}).set_requires_grad(true),
        torch::ones({3}).set_requires_grad(true), torch::randn({1,4}).set_requires_grad(true),
        torch::randn({2,3,4}).set_requires_grad(true)};
}
Params clone(const Params& p) { Params out;for(auto& x:p)out.push_back(x.detach().clone().set_requires_grad(true));return out; }
void step(Muon& opt,Params& p,int n) {
    for(size_t i=0;i<p.size();++i)p[i].mutable_grad() = torch::sin(torch::arange(p[i].numel(),p[i].options().requires_grad(false)).reshape(p[i].sizes())*.19 + n*.37 + static_cast<double>(i));
    opt.step();opt.zero_grad();
}
std::string save(const Muon& opt,bool legacy=false) {
    torch::serialize::OutputArchive out;
    if(legacy)opt.torch::optim::SGD::save(out);else opt.save(out);
    std::ostringstream stream;out.save_to(stream);return stream.str();
}
void load(Muon& opt,const std::string& bytes) {
    std::istringstream stream(bytes);torch::serialize::InputArchive in;in.load_from(stream,torch::kCPU);opt.load(in);
}
template<class F> void rejects(F fn) { bool rejected=false;try{fn();}catch(const std::exception&){rejected=true;}assert(rejected); }
int main() {
    torch::set_num_threads(2);torch::manual_seed(43);
    auto a=initial();Muon uninterrupted(a,MuonOptions(.002).momentum(.95).nesterov(true));
    for(int n=0;n<7;++n)step(uninterrupted,a,n);
    auto b=clone(a);Muon resumed(b,MuonOptions(.9));load(resumed,save(uninterrupted));
    assert(!resumed.loadedLegacyAdamState && resumed.AdamStateCount()==3);
    for(auto* opt : {&uninterrupted,&resumed}) {
        const auto& options=static_cast<const MuonOptions&>(opt->param_groups()[0].options());
        std::cout<<"RESUME_OPTIONS lr="<<options.lr()<<" momentum="<<options.momentum()<<" nesterov="<<options.nesterov()<<std::endl;
    }
    for(int n=7;n<31;++n){step(uninterrupted,a,n);step(resumed,b,n);for(size_t i=0;i<a.size();++i){ if(!torch::equal(a[i],b[i]))std::cerr<<"RESUME_MISMATCH step="<<n<<" parameter="<<i<<" max_delta="<<(a[i]-b[i]).abs().max().item<double>()<<std::endl;assert(torch::equal(a[i],b[i]));}}
    // Repeated save/load, not only the first step after a checkpoint.
    auto c=clone(b);Muon twice(c,MuonOptions(.7));load(twice,save(resumed));
    for(int n=31;n<45;++n){step(uninterrupted,a,n);step(twice,c,n);for(size_t i=0;i<a.size();++i)assert(torch::equal(a[i],c[i]));}
    auto d=clone(a);Muon legacy(d,MuonOptions(.002).momentum(.95).nesterov(true));load(legacy,save(uninterrupted,true));
    assert(legacy.loadedLegacyAdamState && legacy.AdamStateCount()==0);
    step(legacy,d,45);step(uninterrupted,a,45);
    assert(torch::equal(a[0],d[0]) && torch::equal(a[4],d[4])); // saved matrix moments retained
    assert(!torch::equal(a[1],d[1])); // legacy omission has a measurable update effect
    for(int fault=0;fault<5;++fault){
        torch::serialize::OutputArchive root,adam,entry;uninterrupted.torch::optim::SGD::save(root);
        adam.write("version",torch::tensor(int64_t(fault==4?9:2)));
        adam.write("present",torch::tensor(std::vector<int64_t>{0,1,0,0,0}));
        adam.write("options",torch::tensor(std::vector<double>{.002,.95,0,0,1},torch::kFloat64).reshape({1,5}));
        auto mean=torch::zeros(fault==0?5:4);auto square=torch::ones({4});
        if(fault==1)mean[0]=std::numeric_limits<float>::quiet_NaN();
        if(fault==2)square[0]=-1;
        entry.write("mean",mean);entry.write("square",square);entry.write("step",torch::tensor(int64_t(fault==3?0:4)));
        adam.write("1",entry);root.write("muon_adam",adam);std::ostringstream bytes;root.save_to(bytes);
        auto badParams=clone(a);Muon bad(badParams,MuonOptions(.002));rejects([&]{load(bad,bytes.str());});
    }
    // Fresh and never-updated parameters legitimately have no state.
    auto freshParams=initial();Muon fresh(freshParams,MuonOptions(.002));
    auto freshCopy=clone(freshParams);Muon freshLoaded(freshCopy,MuonOptions(.002));load(freshLoaded,save(fresh));
    assert(freshLoaded.AdamStateCount()==0 && !freshLoaded.loadedLegacyAdamState);
    // The production model loader must propagate missing/corrupt optimizer failures.
    auto dir=std::filesystem::temp_directory_path()/ ("mm-muon-resume-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(dir);
    GGL::ModelConfig cfg(GGL::PartialModelConfig{});cfg.layerSizes={3};cfg.numInputs=2;cfg.numOutputs=2;cfg.optimType=GGL::ModelOptimType::MUON;
    GGL::Model model("checked",cfg,torch::kCPU);model.SetOptimLR(.002);
    model.Forward(torch::ones({2,2}),false).sum().backward();model.StepOptim();model.Save(dir);
    GGL::Model restored("checked",cfg,torch::kCPU);restored.Load(dir,false,true);
    assert(!dynamic_cast<Muon*>(restored.optim)->loadedLegacyAdamState);
    for(int n=0;n<3;++n) {
        auto input=torch::full({2,2},.5+n);
        model.Forward(input,false).square().mean().backward();model.StepOptim();
        restored.Forward(input,false).square().mean().backward();restored.StepOptim();
        assert(torch::equal(model.CopyParams(),restored.CopyParams()));
    }
    model.Save(dir);
    auto weights=model.parameters()[0].detach().clone();
    std::filesystem::remove(model.GetOptimSavePath(dir));
    rejects([&]{model.Load(dir,false,true);});assert(torch::equal(weights,model.parameters()[0]));
    {std::ofstream invalid(model.GetOptimSavePath(dir));invalid<<"invalid archive";}
    rejects([&]{model.Load(dir,false,true);});
    std::filesystem::remove_all(dir);
    std::cout<<"PASS: exact repeated Muon resume, legacy effect, malformed Adam state and full-model failure propagation\n";
}
