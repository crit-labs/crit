// Structured-output constraints: byte vocabularies, finite choices, JSON-object syntax, byte grammars and their
// artifacts, private forks/rewinds, limits, the allowed-set readout and constrained sampling through Sampler. The
// json_corpus and grammar_corpus modes answer the independent Python oracles (tests/core/*_constraint_oracle.py).
#include <crit/constraints.h>
#include <crit/sampling.h>
#include <array>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
using namespace crit::core;

namespace {
void require(bool value, const char* message) { if(!value) throw std::runtime_error(message); }
template<class F> void rejects(F action) {
    try { action(); } catch(const std::exception&) { return; }
    throw std::runtime_error("expected rejection missing");
}
using Ids = std::vector<Token>;
std::shared_ptr<const ByteVocabulary> vocabulary(std::vector<std::string> pieces, Token eos) {
    return std::make_shared<const ByteVocabulary>(std::move(pieces), eos);
}
// Every byte as its own token, then EOS (256).
std::shared_ptr<const ByteVocabulary> byte_vocabulary() {
    std::vector<std::string> pieces;
    for(unsigned int byte = 0; byte < 256; ++byte) pieces.emplace_back(1, static_cast<char>(byte));
    pieces.emplace_back();
    return vocabulary(std::move(pieces), 256);
}
std::shared_ptr<const ChoiceGrammar> choices(std::vector<std::string> values) {
    return std::make_shared<const ChoiceGrammar>(std::span<const std::string>(values));
}
const std::vector<std::string> toy_pieces{"a", "b", "c", "", "ab", "x"};   // EOS = 3

void vocabularies() {
    const auto v = vocabulary({"a", "", "bc", ""}, 1);
    require(v->size() == 4 && v->eos() == 1 && v->piece(2) == "bc", "vocabulary pieces");
    rejects([&] { (void)v->piece(4); });
    rejects([&] { (void)v->piece(-1); });
    rejects([] { ByteVocabulary({}, 0); });
    rejects([] { ByteVocabulary({"a"}, 0); });                    // EOS needs an empty piece
    rejects([] { ByteVocabulary({""}, 1); });
    rejects([] { ByteVocabulary({""}, -1); });
}

void choice_matcher() {
    const auto v = vocabulary(toy_pieces, 3);
    ChoiceMatcher state(choices({"ab", "ac"}), v);
    auto copy = state.fork();
    require(state.allowed() == Ids({0, 4}), "initial choice support: a and ab");
    copy->append(0); copy->append(2); copy->append(3);
    require(copy->ended() && copy->allowed() == Ids{3}, "absorbing EOS law");
    rejects([&] { copy->append(3); });
    copy->rewind(1);
    require(copy->allowed() == Ids({1, 2}) && copy->size() == 1, "rewind restores the frontier");
    rejects([&] { copy->append(5); });
    require(copy->size() == 1 && state.allowed() == Ids({0, 4}), "failed append and forks stay private");
    const auto unicode = vocabulary({"\xc3", "\xa9", ""}, 2);
    ChoiceMatcher utf8(choices({"\xc3\xa9"}), unicode);
    utf8.append(0);
    require(utf8.allowed() == Ids{1}, "a token may split a UTF-8 character");
    utf8.append(1); utf8.append(2);
    require(utf8.ended(), "split UTF-8 choice completes");
    ChoiceMatcher overlap(choices({"a", "ab", ""}), v);
    require(overlap.allowed() == Ids({0, 3, 4}), "empty and overlapping choices allow EOS and continuations");
    overlap.append(0);
    require(overlap.allowed() == Ids({1, 3}), "a complete prefix allows both EOS and continuation");
    ChoiceMatcher dead(choices({"q"}), v);
    rejects([&] { (void)dead.allowed(); });                       // no piece spells q: a dead end, never empty
    rejects([] { ChoiceGrammar(std::span<const std::string>{}); });
}

bool json_accepts(const std::string& bytes) {
    try {
        JsonObjectMatcher matcher(byte_vocabulary());
        for(const char raw:bytes) matcher.append(static_cast<unsigned char>(raw));
        matcher.append(256);
        return matcher.ended();
    } catch(const std::exception&) { return false; }
}
void json_matcher() {
    for(const std::string input:{"{}", " \n{\"\":[]}\t", "{\"x\":{\"a\":[true,false,null,-0,1.2e-3]}}",
            "{\"s\":\"\\uD800\\n\\t\\\\\\\"\"}", "{\"s\":\"\xc3\xa9\xf0\x9f\x98\x80\"}", "{\"n\":1e999}"})
        require(json_accepts(input), "valid JSON object rejected");
    for(const std::string input:{"", "[]", "true", "{x:0}", "{\"x\":01}", "{\"x\":-.1}", "{\"x\":1.}", "{\"x\":1e+}",
            "{\"x\":true,}", "{\"x\":[0,]}", "{\"x\":tru}", "{\"x\":null}{}", "{\"x\":\"\\v\"}", "{\"x\":\"\n\"}",
            "{\"x\":\"\xc0\xaf\"}", "{\"x\":\"\xed\xa0\x80\"}", "{\"x\":\"\xf4\x90\x80\x80\"}", "{\"x\":\"\xc3\"}",
            "{\"x\":\"\\u00g0\"}"})
        require(!json_accepts(input), "malformed JSON object accepted");
    const auto v = byte_vocabulary();
    JsonObjectMatcher state(v);
    const std::string prefix = "{\"x\":\"\xc3";
    for(const char raw:prefix) state.append(static_cast<unsigned char>(raw));
    auto branch = state.fork();
    branch->append(0xa9); branch->append('"'); branch->append('}'); branch->append(256);
    require(branch->ended() && !state.ended(), "JSON forks stay private");
    branch->rewind(prefix.size());
    rejects([&] { branch->append('"'); });
    branch->append(0xb1); branch->append('"'); branch->append('}');
    require(dynamic_cast<JsonObjectMatcher&>(*branch).complete(), "complete object");
    branch->append(256);
    require(branch->allowed() == Ids{256}, "terminal JSON law");
    rejects([&] { branch->append(256); });
    rejects([&] { state.append(256); });
    JsonObjectMatcher depth(v, JsonLimits{1, 100, 100});
    for(const char c:std::string("{\"x\":")) depth.append(static_cast<unsigned char>(c));
    rejects([&] { depth.append('['); });
    depth.append('0'); depth.append('}'); depth.append(256);
    JsonObjectMatcher bytes(v, JsonLimits{8, 2, 8});
    bytes.append('{'); bytes.append('}');
    rejects([&] { bytes.append(' '); });
    bytes.append(256);
    JsonObjectMatcher tokens(v, JsonLimits{8, 100, 2});
    tokens.append('{'); tokens.append('}');
    rejects([&] { tokens.append(256); });
    rejects([&] { (void)tokens.allowed(); });
    rejects([&] { JsonObjectMatcher(v, JsonLimits{0, 1, 1}); });
    const auto packed_vocabulary = vocabulary({"{\"a\":[-1.2e", "+3,true]}", "", "x"}, 2);
    JsonObjectMatcher packed(packed_vocabulary);
    packed.append(0);
    auto packed_branch = packed.fork();
    packed_branch->append(1); packed_branch->append(2);
    require(packed_branch->ended(), "one token may cross several JSON structures");
    packed_branch->rewind(1);
    require(packed_branch->allowed() == Ids{1}, "number frontier survives rollback");
}
void json_corpus() {
    // Hex lines avoid any locale conversion between this binary and the oracle.
    std::string line;
    while(std::getline(std::cin, line)) {
        if(line.size() % 2) throw std::invalid_argument("odd hex corpus line");
        std::string bytes;
        for(std::size_t i = 0; i < line.size(); i += 2) bytes.push_back(static_cast<char>(std::stoul(line.substr(i, 2), nullptr, 16)));
        std::cout << (json_accepts(bytes) ? '1' : '0') << '\n';
    }
}

std::shared_ptr<const ByteGrammar> ab_or_ac() {
    return std::make_shared<const ByteGrammar>(2, 0, std::vector<ByteGrammar::Production>{{0, {97, 257}}, {1, {98}}, {1, {99}}});
}
bool grammar_accepts(std::shared_ptr<const ByteGrammar> rules, std::string_view value) {
    GrammarMatcher matcher(std::move(rules), byte_vocabulary(), GrammarLimits{1048576, 1048576, 65536, 4194304});
    try {
        for(const unsigned char c:value) matcher.append(c);
        matcher.append(256);
        return matcher.ended();
    } catch(const std::domain_error&) { return false; }
}
void grammar_matcher() {
    const auto v = vocabulary(toy_pieces, 3);
    GrammarMatcher matcher(ab_or_ac(), v);
    require(matcher.allowed() == Ids({0, 4}), "grammar initial support");
    auto branch = matcher.fork();
    branch->append(0); branch->append(2); branch->append(3);
    require(branch->ended() && branch->allowed() == Ids{3}, "grammar absorbing EOS");
    rejects([&] { branch->append(3); });
    branch->rewind(1);
    require(branch->allowed() == Ids({1, 2}) && matcher.size() == 0, "grammar private rollback");
    rejects([&] { branch->append(5); });
    require(branch->size() == 1, "failed grammar append keeps state");
    const auto recursive = std::make_shared<const ByteGrammar>(1, 0, std::vector<ByteGrammar::Production>{{0, {}}, {0, {40, 256, 41, 256}}});
    for(const auto* text:{"", "()", "(())()", "(()(()))"}) require(grammar_accepts(recursive, text), "recursive grammar rejected");
    for(const auto* text:{"(", ")", "(()", "()x", "())("}) require(!grammar_accepts(recursive, text), "recursive grammar escaped");
    const auto left = std::make_shared<const ByteGrammar>(2, 0,
        std::vector<ByteGrammar::Production>{{0, {256, 97}}, {0, {257}}, {1, {}}, {1, {257}}});
    require(grammar_accepts(left, "aaa") && grammar_accepts(left, ""), "nullable left recursion");
    const auto raw = ByteGrammar::load("crit-byte-grammar-v1 1 2 0\n0 3 195 169 0\n0 0\n");
    require(grammar_accepts(raw, std::string("\xc3\xa9\0", 3)), "split UTF-8 and embedded NUL");
    require(!grammar_accepts(raw, "\xc3"), "truncated UTF-8 literal");
    for(const auto* text:{"", "crit-byte-grammar-v0 1 1 0", "crit-byte-grammar-v1 1 1 0 0 1 256",
                          "crit-byte-grammar-v1 1 1 0 0 1 257", "crit-byte-grammar-v1 1 1 0 0 0 garbage",
                          "crit-byte-grammar-v1 1 1 0 0 1 -1", "crit-byte-grammar-v1 2 1 0 0 0"})
        rejects([&] { (void)ByteGrammar::load(text); });
    rejects([] { (void)ByteGrammar::load(std::string((std::size_t{4} << 20) + 1, ' ')); });
    GrammarMatcher small(ab_or_ac(), v, GrammarLimits{4, 4, 1, 1});
    rejects([&] { small.append(0); });
    require(small.size() == 0, "chart exhaustion leaves the matcher unchanged");
    GrammarMatcher limited(ab_or_ac(), v, GrammarLimits{1, 3, 4096, 262144});
    limited.append(0);
    rejects([&] { limited.append(1); });
    require(limited.size() == 1, "byte limit keeps the prefix");
    auto dead = limited.fork();
    dead->rewind(0);
    require(dead->size() == 0, "a dead-end fork can still rewind");
    rejects([&] { GrammarMatcher(ab_or_ac(), v, GrammarLimits{0, 1, 1, 1}); });
    // Seeded walks: the compiled grammar and an independent finite-choice matcher allow the same tokens everywhere.
    const auto words = vocabulary({"a", "b", "c", "", "ab", "ba", "abc", "cab"}, 3);
    const auto grammar = ByteGrammar::load("crit-byte-grammar-v1 3 6 0\n0 2 257 258\n1 1 97\n1 2 97 98\n2 1 99\n2 2 98 99\n2 0\n");
    std::vector<std::string> language;
    for(const char* head:{"a", "ab"}) for(const char* tail:{"c", "bc", ""}) language.push_back(std::string(head) + tail);
    std::uint64_t state = 12345;
    for(int walk = 0; walk < 400; ++walk) {
        GrammarMatcher g(grammar, words);
        ChoiceMatcher c(choices(language), words);
        while(!g.ended()) {
            const auto allowed = g.allowed();
            require(allowed == c.allowed(), "compiled grammar and choice oracle disagree");
            state = state*6364136223846793005ULL + 1442695040888963407ULL;
            const Token token = allowed[(state >> 33) % allowed.size()];
            g.append(token); c.append(token);
            if(state >> 63 && g.size() > 1) { g.rewind(g.size() - 1); c.rewind(c.size() - 1); }
        }
        require(c.ended(), "EOS agrees with the choice oracle");
    }
}
void grammar_corpus(const char* path) {
    std::ifstream input(path, std::ios::binary);
    std::string text((std::size_t{4} << 20) + 1, '\0');
    input.read(text.data(), static_cast<std::streamsize>(text.size()));
    text.resize(static_cast<std::size_t>(input.gcount()));
    const auto rules = ByteGrammar::load(text);
    std::string line;
    while(std::getline(std::cin, line)) {
        if(line.size() % 2) throw std::invalid_argument("odd grammar corpus line");
        std::string bytes;
        for(std::size_t i = 0; i < line.size(); i += 2) bytes.push_back(static_cast<char>(std::stoul(line.substr(i, 2), nullptr, 16)));
        std::cout << (grammar_accepts(rules, bytes) ? '1' : '0') << '\n';
    }
}

void readout() {
    const std::vector<float> scores{100, 0, 2, 2, -100};
    const Ids ids{1, 2, 3};
    const auto r = constraint_readout(scores, ids);
    require(r.ids == ids && r.original_mass < 1e-40 && r.probabilities[1] == r.probabilities[2], "readout mass and ties");
    const double total = 1 + 2*std::exp(2.0);
    require(std::abs(r.probabilities[0] - 1/total) < 1e-15, "restricted softmax");
    const std::vector<float> even{0, 0, 0, 0};
    require(std::abs(constraint_readout(even, Ids{0, 3}).original_mass - 0.5) < 1e-15, "original mass");
    const float minus_inf = -std::numeric_limits<float>::infinity();
    const std::vector<float> masked{minus_inf, 1, 1};
    require(constraint_readout(masked, Ids{0, 1}).probabilities == std::vector<double>({0, 1}), "masked logits");
    rejects([&] { (void)constraint_readout(masked, Ids{0}); });
    rejects([&] { (void)constraint_readout(scores, Ids{2, 1}); });
    rejects([&] { (void)constraint_readout(scores, Ids{2, 2}); });
    rejects([&] { (void)constraint_readout(scores, Ids{5}); });
    rejects([&] { (void)constraint_readout(scores, Ids{}); });
    auto bad = scores;
    bad[0] = std::numeric_limits<float>::infinity();
    rejects([&] { (void)constraint_readout(bad, ids); });
}

// A toy causal model: logits depend only on the last token. Sampling through the matcher's allowed set must stay in
// the language and give each complete answer its causal probability (products of the restricted softmax).
void sampling() {
    const std::vector<std::string> pieces{"{\"x\":", "true}", "false}", "", "{}", "invalid"};
    const auto v = vocabulary(pieces, 3);
    const auto logits = [](const Ids& output) {
        std::vector<float> row{1, 3, 0, 2, 4, 99};
        if(!output.empty() && output.back() == 0) { row[0] = -1000; row[4] = -1000; }
        return row;
    };
    // Independent answer probabilities: first {0, 4} (logits 1 and 4), then {1, 2} after token 0 (logits 3 and 0).
    const double first = 1/(1 + std::exp(3.0)), yes = 1/(1 + std::exp(-3.0));
    const std::map<std::string, double> expected{{"{}", 1 - first}, {"{\"x\":true}", first*yes}, {"{\"x\":false}", first*(1 - yes)}};
    std::map<std::string, std::size_t> counts;
    Sampler sampler(6);
    constexpr int runs = 4000;
    for(int seed = 0; seed < runs; ++seed) {
        JsonObjectMatcher matcher(v);
        SamplingParams params;
        params.seed = std::uint64_t(seed);
        TokenHistory history({});
        Ids output;
        std::string answer;
        while(!matcher.ended()) {
            const auto allowed = matcher.allowed();
            const Token token = sampler.sample(logits(output), AllowedTokens{allowed}, params, history).token;
            matcher.append(token);
            history.stage(std::span<const Token>(&token, 1));
            history.commit(1);
            output.push_back(token);
            if(token != 3) answer += pieces[std::size_t(token)];
        }
        require(json_accepts(answer), "constrained sampling left the language");
        ++counts[answer];
    }
    require(counts.size() == expected.size(), "unexpected constrained answers");
    for(const auto& [answer, p]:expected)
        require(std::abs(double(counts[answer])/runs - p) < 0.025, "constraint changed the causal answer law");
}
}

int main(int argc, char** argv) {
    try {
        const std::string_view name = argc > 1 ? argv[1] : "all";
        if(name == "json_corpus") { json_corpus(); return 0; }
        if(name == "grammar_corpus" && argc == 3) { grammar_corpus(argv[2]); return 0; }
        const std::vector<std::pair<std::string_view, std::function<void()>>> cases{
            {"constraint_vocabulary", vocabularies}, {"constraint_choices", choice_matcher},
            {"constraint_json", json_matcher}, {"constraint_grammar", grammar_matcher},
            {"constraint_readout", readout}, {"constraint_sampling", sampling}};
        bool found = false;
        for(const auto& [case_name, run]:cases) if(name == "all" || name == case_name) {
            run();
            found = true;
            std::cout << "crit core constraints passed: " << case_name << '\n';
        }
        require(found, "unknown test case");
    } catch(const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
