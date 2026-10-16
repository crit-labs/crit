// Synthetic tokenizer/UTF-8 fragments; exercises the public lifetime boundary.
#include <crit/core.h>
#include <condition_variable>
#include <future>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
using namespace crit::core;
namespace {
void require(bool value, const char* message) { if(!value) throw std::runtime_error(message); }
template<class F> void rejects(F action, std::string_view expected) {
    try { action(); } catch(const std::exception& e) {
        require(std::string_view(e.what()).find(expected)!=std::string_view::npos,"unexpected diagnostic");
        return;
    }
    throw std::runtime_error("expected rejection missing");
}
struct Gate {
    std::mutex mutex;
    std::condition_variable cv;
    bool entered=false, released=false;
    void wait() {
        std::unique_lock lock(mutex); entered=true; cv.notify_all();
        require(cv.wait_for(lock,std::chrono::seconds(5),[&]{return released;}),"fixture gate expired");
    }
    void observed() {
        std::unique_lock lock(mutex);
        require(cv.wait_for(lock,std::chrono::seconds(5),[&]{return entered;}),"worker did not reach gate");
    }
    void release() { std::lock_guard lock(mutex); released=true; cv.notify_all(); }
};
struct Release { std::shared_ptr<Gate> gate; ~Release() {gate->release();} };
struct Frontend final:TextFrontend {
    bool null_decoder=false;
    struct Decoder final:TextDecoder {
        const Frontend& owner;
        std::string pending;
        explicit Decoder(const Frontend& f):owner(f) {}
        std::string push(std::span<const Token> tokens) override {
            std::string result;
            for(auto token:tokens) {
                if(token==255) throw std::runtime_error("synthetic decoder failure");
                pending+=owner.detokenize({&token,1});
                // This fixture recognizes ASCII and one split UTF-8 character.
                if(pending=="\xc3\xa9" || (pending.size()==1 && token<128))
                    result+=std::exchange(pending,{});
            }
            return result;
        }
        std::string finish() override {
            if(!pending.empty()) throw std::runtime_error("incomplete text");
            return {};
        }
    };
    std::vector<Token> tokenize(std::string_view text) const override {
        if(text=="invalid") return {256};
        std::vector<Token> result;
        for(unsigned char byte:text) result.push_back(byte);
        return result;
    }
    std::string detokenize(std::span<const Token> tokens) const override {
        std::string result;
        for(auto token:tokens) result.push_back(static_cast<char>(token));
        return result;
    }
    std::unique_ptr<TextDecoder> create_decoder() const override {
        return null_decoder ? nullptr : std::make_unique<Decoder>(*this);
    }
};
struct Host final:Backend {void bind_thread() override {} void drain() override {}};
struct Turn final:Session {
    std::shared_ptr<Gate> gate;
    Token output=0;
    int index=0;
    explicit Turn(std::shared_ptr<Gate> g):gate(std::move(g)) {}
    Progress advance(WorkBudget budget) override {
        if(index==1) gate->wait();
        output=index++ ? 169 : 195;
        return {budget.prefill_tokens,true,{&output,1},nullptr};
    }
    void commit(std::uint32_t,bool) override {}
    void close(Finish) noexcept override {}
};
struct Adapter final:Model {
    std::shared_ptr<const TextFrontend> frontend;
    std::shared_ptr<Gate> gate;
    Adapter(std::shared_ptr<const TextFrontend> f,std::shared_ptr<Gate> g):frontend(std::move(f)),gate(std::move(g)) {}
    ModelInfo info() const override {return {"frontend.fixture",256,8,2,{},true,false};}
    std::unique_ptr<Session> open(std::span<const Token>) override {return std::make_unique<Turn>(gate);}
    std::vector<Token> tokenize(std::string_view) override {return {42};}
    std::string detokenize(std::span<const Token>) override {return "legacy";}
    std::shared_ptr<const TextFrontend> text_frontend() const override {return frontend;}
};
std::unique_ptr<Engine> make(std::shared_ptr<const TextFrontend> frontend,std::shared_ptr<Gate> gate) {
    return std::make_unique<Engine>(Profile{8,2,4096,4096},std::make_shared<Host>(),
        [frontend,gate](const Profile&,Services){return std::make_unique<Adapter>(frontend,gate);});
}
void independent() {
    auto frontend=std::make_shared<Frontend>();
    std::weak_ptr<const TextFrontend> lifetime=frontend;
    auto gate=std::make_shared<Gate>();
    auto engine=make(frontend,gate);
    frontend.reset();
    auto decoder=engine->create_decoder(), other=engine->create_decoder();
    auto request=engine->submit({65},2);
    Release release{gate};
    gate->observed();
    std::vector<std::future<void>> callers;
    for(int i=0;i<8;++i) callers.push_back(std::async(std::launch::async,[&] {
        for(int j=0;j<100;++j) {
            auto tokens=engine->tokenize("xyz");
            require(engine->detokenize(tokens)=="xyz","busy frontend changed text");
        }
    }));
    for(auto& caller:callers) caller.get();
    std::string text;
    auto result=request.wait([&](Token token) {
        require(engine->detokenize(engine->tokenize("callback"))=="callback","token sink cannot access frontend");
        text+=decoder->push({&token,1});
        if(token==195) {require(text.empty(),"partial UTF-8 escaped decoder"); gate->release();}
    });
    require(result.tokens==std::vector<Token>({195,169}) && text=="\xc3\xa9","stream output differs");
    require(decoder->finish().empty(),"unexpected trailing text");
    rejects([&]{decoder->finish();},"closed");
    Token invalid=-1, lead=195, tail=169;
    rejects([&]{other->push({&invalid,1});},"vocabulary");
    rejects([&]{engine->tokenize("invalid");},"vocabulary");
    rejects([&]{engine->detokenize({&invalid,1});},"vocabulary");
    engine.reset(); decoder.reset();
    require(!lifetime.expired(),"decoder dropped borrowed frontend tables");
    require(other->push({&lead,1}).empty(),"private decoder inherited another stream");
    require(other->push({&tail,1})=="\xc3\xa9","decoder unusable after engine destruction");
    other->finish(); other.reset();
    require(lifetime.expired(),"frontend lifetime leaked");
}
void failures_and_legacy() {
    auto frontend=std::make_shared<Frontend>();
    auto gate=std::make_shared<Gate>();
    auto engine=make(frontend,gate);
    auto decoder=engine->create_decoder();
    Token bad=255, good=65;
    rejects([&]{decoder->push({&bad,1});},"decoder failure");
    rejects([&]{decoder->push({&good,1});},"closed");
    require(engine->tokenize("a")==std::vector<Token>{97},"private decoder failure affected frontend");
    frontend->null_decoder=true;
    rejects([&]{engine->create_decoder();},"null decoder");
    auto legacy=make({},gate);
    rejects([&]{legacy->create_decoder();},"independent text frontend");
    auto request=legacy->submit({65},2);
    Release release{gate}; gate->observed();
    rejects([&]{legacy->tokenize("a");},"unavailable");
    rejects([&]{legacy->detokenize({&good,1});},"unavailable");
    gate->release(); request.wait();
    require(legacy->tokenize("a")==std::vector<Token>{42},"legacy idle tokenization changed");
}
// Byte tokens without an incremental decoder; an incomplete trailing UTF-8 sequence decodes as U+FFFD.
struct Bytes final:TextFrontend {
    std::vector<Token> tokenize(std::string_view text) const override {
        std::vector<Token> result;
        for(unsigned char byte:text) result.push_back(byte);
        return result;
    }
    std::string detokenize(std::span<const Token> tokens) const override {
        std::string result;
        for(auto token:tokens) result.push_back(static_cast<char>(token));
        auto start=result.size();
        while(start>0 && result.size()-start<3 && (static_cast<unsigned char>(result[start-1])&0xC0)==0x80) --start;
        if(start>0) {
            const auto lead=static_cast<unsigned char>(result[start-1]);
            const std::size_t need=lead>=0xF0 ? 4 : lead>=0xE0 ? 3 : lead>=0xC0 ? 2 : 1;
            if(result.size()-(start-1)<need) result.replace(start-1,std::string::npos,"\xEF\xBF\xBD");
        }
        return result;
    }
};
// SentencePiece-like words without an incremental decoder: 1 " hello", 2 " world", 3 " " U+FFFD, and the decoded text
// loses one leading space, so a continuation keeps its space only when decoded after earlier text.
struct Spaced final:TextFrontend {
    std::vector<Token> tokenize(std::string_view) const override {return {};}
    std::string detokenize(std::span<const Token> tokens) const override {
        std::string result;
        for(auto token:tokens) result+=token==1 ? " hello" : token==2 ? " world" : token==3 ? " \xEF\xBF\xBD" : "";
        if(!result.empty() && result.front()==' ') result.erase(0,1);
        return result;
    }
};
// A frontend whose decoder construction fails for its own reasons; Core must not fall back.
struct Broken final:TextFrontend {
    std::vector<Token> tokenize(std::string_view) const override {return {};}
    std::string detokenize(std::span<const Token>) const override {return {};}
    std::unique_ptr<TextDecoder> create_decoder() const override {throw std::domain_error("decoder tables missing");}
};
// Serial text: an adapter that emits scripted token groups (the last group repeats).
struct Script {
    std::vector<std::vector<Token>> steps;
    std::vector<std::pair<std::uint32_t,bool>> commits;
    Finish closed=Finish::length;
    int opens=0;
};
struct ScriptTurn final:Session {
    std::shared_ptr<Script> script;
    std::size_t step=0;
    std::vector<Token> out;
    explicit ScriptTurn(std::shared_ptr<Script> s):script(std::move(s)) {++script->opens;}
    Progress advance(WorkBudget budget) override {
        out=script->steps.at(std::min(step++,script->steps.size()-1));
        if(out.size()>budget.output_tokens) out.resize(budget.output_tokens);
        return {budget.prefill_tokens,true,out,nullptr};
    }
    void commit(std::uint32_t accepted,bool terminal) override {
        script->commits.push_back({accepted,terminal});
        std::fill(out.begin(),out.end(),0); // A physical commit may invalidate the borrowed tokens.
    }
    void close(Finish finish) noexcept override {script->closed=finish;}
};
struct TextAdapter final:Model {
    std::shared_ptr<const TextFrontend> frontend;
    std::shared_ptr<Script> script;
    TextAdapter(std::shared_ptr<const TextFrontend> f,std::shared_ptr<Script> s):frontend(std::move(f)),script(std::move(s)) {}
    ModelInfo info() const override {return {"frontend.text.fixture",256,8,2,{},true,false};}
    std::unique_ptr<Session> open(std::span<const Token>) override {return std::make_unique<ScriptTurn>(script);}
    std::unique_ptr<Session> open_request(std::span<const Token>,const RequestOptions&) override {return std::make_unique<ScriptTurn>(script);}
    std::shared_ptr<const TextFrontend> text_frontend() const override {return frontend;}
};
std::unique_ptr<Engine> make_text(std::shared_ptr<const TextFrontend> frontend,std::shared_ptr<Script> script) {
    return std::make_unique<Engine>(Profile{8,2,4096,4096},std::make_shared<Host>(),
        [frontend,script](const Profile&,Services){return std::make_unique<TextAdapter>(frontend,script);});
}
using Ends=std::vector<std::uint32_t>;
using Tokens=std::vector<Token>;
void serial_text() {
    auto frontend=std::make_shared<Frontend>(); auto script=std::make_shared<Script>();
    auto engine=make_text(frontend,script);
    RequestOptions text; text.text=true;
    // Each token's released text, held while a character is incomplete.
    script->steps={{65},{195},{169},{66}};
    std::vector<std::string> pieces;
    auto result=engine->submit({72},4,text).wait_output([&](const OutputEvent& event){pieces.emplace_back(event.text);});
    require(result.text=="A\xc3\xa9" "B" && result.text_ends==Ends({1,1,3,4}),"serial text offsets");
    require(pieces==std::vector<std::string>({"A","","\xc3\xa9","B"}),"output event text pieces");
    script->steps={{65,195,169,66}};
    result=engine->submit({72},4,text).wait();
    require(result.text=="A\xc3\xa9" "B" && result.text_ends==Ends({1,1,3,4}),"multi-token step offsets");
    // Stop strings across token boundaries and inside one multi-token delivery; the completing token is kept.
    RequestOptions stop; stop.stop_strings={"BC"};
    script->steps={{65},{66},{67},{68}}; script->commits.clear();
    result=engine->submit({72},6,stop).wait();
    require(result.tokens==Tokens({65,66,67}) && result.finish==Finish::stop && result.text=="A" && result.text_ends==Ends({1,1,1}),
            "stop string across tokens");
    require(script->commits.back()==std::make_pair(1u,true) && script->closed==Finish::stop,"stop string step commit");
    script->steps={{65,66,67,68}}; script->commits.clear();
    result=engine->submit({72},6,stop).wait();
    require(result.tokens==Tokens({65,66,67}) && result.text=="A" &&
            script->commits==std::vector<std::pair<std::uint32_t,bool>>({{3,true}}),"stop string inside a delivery");
    // A stop token is kept but never decoded (this decoder would hold its byte and fail at finish).
    RequestOptions token_stop; token_stop.text=true; token_stop.stop_tokens=std::vector<Token>{200};
    script->steps={{65,200,66}};
    result=engine->submit({72},6,token_stop).wait();
    require(result.tokens==Tokens({65,200}) && result.finish==Finish::stop && result.text=="A" && result.text_ends==Ends({1,1}),
            "stop token was decoded");
    // Text held for a possible stop is flushed when the output budget ends the request.
    RequestOptions held; held.stop_strings={"BX"};
    script->steps={{65},{66}};
    result=engine->submit({72},2,held).wait();
    require(result.text=="AB" && result.text_ends==Ends({1,2}) && result.finish==Finish::length,"held text not flushed at length");
    // Decoder failures fail only that request, in push or in the final flush.
    script->steps={{65},{255},{66}};
    std::vector<Token> delivered;
    rejects([&]{engine->submit({72},3,text).wait([&](Token token){delivered.push_back(token);});},"synthetic decoder failure");
    require(delivered==Tokens({65}) && script->closed==Finish::failed,"decoder failure published its step");
    script->steps={{65},{195}};
    rejects([&]{engine->submit({72},2,text).wait();},"incomplete text");
    script->steps={{65}};
    require(engine->submit({72},1,text).wait().text=="A" && engine->usage(Memory::host).peak<4096,"decoder failure poisoned the engine");
    // Decoder construction errors reject the submission without opening a session.
    const auto opens=script->opens;
    frontend->null_decoder=true;
    rejects([&]{engine->submit({72},1,text);},"null decoder");
    frontend->null_decoder=false;
    auto broken=make_text(std::make_shared<Broken>(),script);
    rejects([&]{broken->submit({72},1,stop);},"decoder tables missing");
    auto legacy=make({},std::make_shared<Gate>());
    rejects([&]{legacy->submit({72},1,text);},"independent text frontend");
    rejects([&]{legacy->submit({72},1,stop);},"independent text frontend");
    require(script->opens==opens && broken->usage(Memory::host).live==0,"rejected text submission opened or retained state");
}
void fallback_text() {
    // Without create_decoder, Core decodes with offset_decoder seeded by the prompt tail.
    auto script=std::make_shared<Script>();
    auto engine=make_text(std::make_shared<Bytes>(),script);
    RequestOptions text; text.text=true;
    script->steps={{195},{169},{66}};
    std::vector<std::string> pieces;
    auto result=engine->submit({72,73},3,text).wait_output([&](const OutputEvent& event){pieces.emplace_back(event.text);});
    require(result.text=="\xc3\xa9" "B" && pieces==std::vector<std::string>({"","\xc3\xa9","B"}),"offset decoder fallback");
    // The prompt tail seeds the decoder: a continuation keeps the leading space a context-free decode would strip,
    // also after a prompt ending with a literal U+FFFD.
    auto spaced=make_text(std::make_shared<Spaced>(),script);
    script->steps={{2}};
    for(const auto& prompt:{Tokens{1},Tokens{2,1},Tokens{1,3}}) {
        result=spaced->submit(prompt,1,text).wait();
        require(result.text==" world" && result.text_ends==Ends({6}),"offset decoder fallback lost the prompt context");
    }
    script->steps={{1},{2}};
    result=spaced->submit({1},2,text).wait();
    require(result.text==" hello world" && result.text_ends==Ends({6,12}),"offset decoder fallback continuation");
    // A stop may complete in the flushed tail: the incomplete character flushes as U+FFFD.
    RequestOptions tail; tail.stop_strings={"\xEF\xBF\xBD"};
    script->steps={{65},{195}};
    result=engine->submit({72},2,tail).wait();
    require(result.finish==Finish::stop && result.text=="A" && result.tokens==Tokens({65,195}) && result.text_ends==Ends({1,1}),
            "stop in the flushed tail");
    // The same incomplete tail without a stop is flushed into the last token's text.
    result=engine->submit({72},2,text).wait();
    require(result.finish==Finish::length && result.text=="A\xEF\xBF\xBD" && result.text_ends==Ends({1,4}),"flushed tail text");
}
}
int main() {
    try {independent(); failures_and_legacy(); serial_text(); fallback_text(); std::cout<<"Core text frontend contracts passed\n";}
    catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
