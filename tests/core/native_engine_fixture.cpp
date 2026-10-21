// A toy native engine exported through crit/native_engine.h, for the ABI and crit_core.native tests. Generation
// emits two tokens per decode step after the prompt (a stand-in for speculative rounds), one per step for a
// PlainDecoding request, or the tokens of a ReplayContinuation; scoring returns fixed log-probabilities and the next prompt token as argmax.
// With logprobs n, token t has log-probability -0.5 and the alternatives t, t+1, ... with -0.5, -0.75, ...
#include "crit/json.h"
#include "crit/native_engine.h"
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <chrono>
#include <vector>

using namespace crit::core;

namespace {
constexpr std::uint32_t context_tokens = 256;
constexpr Token stop_token = 5;

struct ToyBackend final : Backend {
    void bind_thread() override {}
    void drain() override {}
};

struct ToySession final : Session {
    std::vector<Token> prompt;
    std::vector<Token> replay;
    std::uint32_t frontier = 0;
    std::uint32_t produced = 0;
    bool plain = false;
    Token base = 0;
    Token emitted[2] = {0, 0};
    std::optional<std::uint32_t> alternatives;  // RequestOptions::logprobs
    float chosen[2] = {0, 0};
    std::vector<TokenLogprob> top;
    Token next_token() {
        const Token t = replay.empty() ? Token(10 + (base + Token(produced)) % 50) : replay.at(produced);
        ++produced;
        return t;
    }
    Progress rows(Progress p) {
        if(!alternatives) return p;
        top.clear();
        for(std::size_t i = 0; i < p.tokens.size(); ++i) {
            chosen[i] = -0.5F;
            for(std::uint32_t k = 0; k < *alternatives; ++k)
                top.push_back({Token((p.tokens[i] + Token(k)) % 64), -0.5F - 0.25F * float(k)});
        }
        p.logprobs = {chosen, p.tokens.size()};
        p.top_logprobs = top;
        return p;
    }
    Progress advance(WorkBudget budget) override {
        if(frontier < prompt.size()) {
            const auto count = std::min<std::uint32_t>(budget.prefill_tokens, std::uint32_t(prompt.size()) - frontier);
            frontier += count;
            if(frontier < prompt.size()) return {count, false, {}};
            emitted[0] = next_token();
            return rows({count, true, {emitted, 1}});
        }
        const auto room = std::min<std::uint32_t>(plain ? 1 : 2, budget.output_tokens);
        std::uint32_t n = 0;
        while(n < room) emitted[n++] = next_token();
        return rows({0, true, {emitted, n}});
    }
    void commit(std::uint32_t, bool) override {}
    void close(Finish) noexcept override {}
};

struct ToyModel final : Model {
    ModelInfo info() const override {
        ModelInfo m;
        m.implementation = "toy native engine";
        m.vocabulary = m.valid_vocabulary = 64;
        m.context = context_tokens;
        m.prefill_chunk = 16;
        m.stop_tokens = {stop_token};
        m.scoring = m.query_scoring = true;
        m.request_context = true;
        m.token_logprobs = true;
        m.max_top_logprobs = 20;
        m.score_paths = {"prefill", "decode"};
        return m;
    }
    std::unique_ptr<Session> open(std::span<const Token> prompt) override { return open_request(prompt, {}); }
    std::unique_ptr<Session> open_request(std::span<const Token> prompt, const RequestOptions& options) override {
        auto session = std::make_unique<ToySession>();
        session->prompt.assign(prompt.begin(), prompt.end());
        for(const auto t:prompt) session->base += t;
        if(const auto* replay = dynamic_cast<const ReplayContinuation*>(options.context.get())) session->replay = replay->tokens;
        session->plain = dynamic_cast<const PlainDecoding*>(options.context.get()) != nullptr;
        session->alternatives = options.logprobs;
        return session;
    }
    std::vector<float> score(std::span<const Token> ids, std::uint32_t first) override {
        return std::vector<float>(ids.size() - first, -0.5F);
    }
    QueryScores score_queries(std::span<const Token> ids, std::uint32_t first, const ScoreQueries& queries,
                              const RequestOptions&) override {
        QueryScores out;
        const float shift = queries.path == "decode" ? -0.25F : 0.F;
        out.target.assign(ids.size() - first, -0.5F + shift);
        for(std::size_t i = 0; i < queries.ids.size(); ++i) out.queries.push_back(-1.F - 0.125F * float(i % 8));
        for(std::size_t t = first; t < ids.size(); ++t) out.argmax.push_back(ids[t]);
        return out;
    }
};

struct BatchedSession final : ConcurrentSession {
    ToySession toy;
    void close(Finish) noexcept override {}
};
struct Settled final : Retirement { bool ready() override { return true; } };
struct BatchedModel final : ConcurrentModel {
    ModelInfo info() const override {
        auto m = ToyModel().info();
        m.max_concurrency = 8; // Deliberately greater than the configured capacity.
        m.scoring = m.query_scoring = false;
        m.token_logprobs = false;
        m.max_top_logprobs = 0;
        m.score_paths.clear();
        return m;
    }
    std::unique_ptr<ConcurrentSession> open(const RequestView& view) override {
        auto s = std::make_unique<BatchedSession>();
        s->toy.prompt.assign(view.prompt.begin(), view.prompt.end());
        for(auto t:view.prompt) s->toy.base += t;
        return s;
    }
    RowProgress prefill(const WorkItem& item) override {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        return {item.key,static_cast<BatchedSession*>(item.session)->toy.advance(item.budget)};
    }
    std::vector<RowProgress> decode(std::span<const WorkItem> items) override {
        std::vector<RowProgress> rows;
        for(const auto& item:items) rows.push_back({item.key,static_cast<BatchedSession*>(item.session)->toy.advance(item.budget)});
        return rows;
    }
    void commit(std::span<const CommitDecision>) override {}
    std::unique_ptr<Retirement> retire(RequestKey,ConcurrentSession&,Finish) override {
        return std::make_unique<Settled>();
    }
    void quiesce() override {}
};

NativeEngine make_toy(std::string_view configuration) {
    const auto document = json::Document::parse(configuration, "toy configuration");
    if(const auto* fail = document.root().find("fail"); fail && fail->boolean()) throw std::runtime_error("toy engine refused to open");
    NativeEngine native;
    native.owner = std::make_shared<ToyBackend>();
    Profile profile{context_tokens, 16, std::size_t(1) << 20, std::size_t(1) << 20};
    if(const auto* concurrent = document.root().find("concurrent"); concurrent && concurrent->boolean()) {
        profile.max_concurrency = 2;
        profile.max_pending = 1;
        native.engine = std::make_unique<Engine>(profile, std::static_pointer_cast<ToyBackend>(native.owner),
                                                [](const Profile&, Services) { return std::make_unique<BatchedModel>(); });
        native.concurrent_requests = true;
        native.result = [](const Result& result) { return "{\"identity\":" + std::to_string(result.request_id) + "}"; };
        return native;
    }
    native.engine = std::make_unique<Engine>(profile, std::static_pointer_cast<Backend>(std::static_pointer_cast<ToyBackend>(native.owner)),
                                             [](const Profile&, Services) { return std::make_unique<ToyModel>(); });
    native.info = [] { return std::string(R"({"speculative": true, "draft_width": 1, "replay": true, "plain": true})"); };
    native.request = [] { return std::string(R"({"accepted": [1, 1]})"); };
    return native;
}
}  // namespace

CRIT_EXPORT_ENGINE(make_toy)
