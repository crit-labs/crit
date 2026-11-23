// Core's structured-output matchers (crit/constraints.h) as crit_core_steps exports them for the host side of Core's
// runtime (crit_core.native): a constrained request's matcher masks each of its tokens before the model plugin's
// sampler draws it, and takes the token Core accepted. One vocabulary per model, one language per constraint (json
// mode, or a compiled crit-byte-grammar-v1 artifact) with the AllowedCache its matchers share, one matcher per
// request. The list of functions is in crit/native_engine.h.
#include "crit/constraints.h"
#include "crit/native_engine.h"
#include <cstdint>
#include <exception>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
using namespace crit::core;
thread_local std::string last;

// Limits for a whole reply: the most output any request decodes (Core's output limit cuts it first), and the
// chart budget of crit_core.constraints.GrammarState, which crit serve re-checks a reply with.
constexpr GrammarLimits reply_grammar{std::size_t{1} << 20, std::size_t{1} << 20, 65536, 4194304};
constexpr JsonLimits reply_json{64, std::size_t{1} << 20, std::size_t{1} << 20};

struct Vocabulary {
    std::shared_ptr<const ByteVocabulary> table;
};
struct Language {
    std::shared_ptr<const ByteVocabulary> vocabulary;
    std::shared_ptr<const ByteGrammar> grammar;   // null: one JSON object
    std::shared_ptr<AllowedCache> cache = std::make_shared<AllowedCache>();
};
struct Matcher {
    std::unique_ptr<TokenMatcher> state;
};

template <class Run> auto guarded(Run&& run, decltype(run()) failed) noexcept -> decltype(run()) {
    try {
        return run();
    } catch(const std::exception& error) {
        try {
            last = error.what();
        } catch(...) {
        }
    } catch(...) {
        try {
            last = "unknown constraint failure";
        } catch(...) {
        }
    }
    return failed;
}
}  // namespace

CRIT_ENGINE_EXPORT void* crit_vocabulary_open(const unsigned char* bytes, const std::uint64_t* ends, std::uint32_t count,
                                              std::int32_t eos) {
    return guarded([&]() -> void* {
        if(!ends || !count) throw std::invalid_argument("a vocabulary needs its pieces");
        std::vector<std::string> pieces(count);
        std::uint64_t start = 0;
        for(std::uint32_t i = 0; i < count; ++i) {
            if(ends[i] < start) throw std::invalid_argument("piece ends must not decrease");
            if(ends[i] > start && !bytes) throw std::invalid_argument("pieces need their bytes");
            pieces[i].assign(reinterpret_cast<const char*>(bytes) + start, std::size_t(ends[i] - start));
            start = ends[i];
        }
        return new Vocabulary{std::make_shared<const ByteVocabulary>(std::move(pieces), Token(eos))};
    }, nullptr);
}

CRIT_ENGINE_EXPORT void crit_vocabulary_close(void* vocabulary) { delete static_cast<Vocabulary*>(vocabulary); }

CRIT_ENGINE_EXPORT void* crit_language_open(void* vocabulary, const char* kind, const char* artifact, std::uint64_t size) {
    return guarded([&]() -> void* {
        if(!vocabulary || !kind) throw std::invalid_argument("a language needs a vocabulary and a kind");
        auto language = std::make_unique<Language>();
        language->vocabulary = static_cast<Vocabulary*>(vocabulary)->table;
        const std::string_view name(kind);
        if(name == "grammar") {
            if(!artifact) throw std::invalid_argument("a grammar language needs its artifact");
            language->grammar = ByteGrammar::load(std::string_view(artifact, std::size_t(size)));
        } else if(name != "json_object") {
            throw std::invalid_argument("a language is json_object or grammar");
        }
        return language.release();
    }, nullptr);
}

CRIT_ENGINE_EXPORT void crit_language_close(void* language) { delete static_cast<Language*>(language); }

CRIT_ENGINE_EXPORT void* crit_matcher_open(void* language) {
    return guarded([&]() -> void* {
        if(!language) throw std::invalid_argument("a matcher needs its language");
        const auto& l = *static_cast<Language*>(language);
        auto matcher = std::make_unique<Matcher>();
        if(l.grammar) matcher->state = std::make_unique<GrammarMatcher>(l.grammar, l.vocabulary, reply_grammar, l.cache);
        else matcher->state = std::make_unique<JsonObjectMatcher>(l.vocabulary, reply_json, l.cache);
        return matcher.release();
    }, nullptr);
}

CRIT_ENGINE_EXPORT int crit_matcher_allowed(void* matcher, std::uint32_t* words, std::uint32_t count) {
    return guarded([&]() -> int {
        if(!matcher || !words) throw std::invalid_argument("allowed needs a matcher and its words");
        static_cast<Matcher*>(matcher)->state->allowed_bits(std::span<std::uint32_t>(words, count));
        return 0;
    }, -1);
}

CRIT_ENGINE_EXPORT int crit_matcher_append(void* matcher, std::int32_t token) {
    return guarded([&]() -> int {
        if(!matcher) throw std::invalid_argument("append needs a matcher");
        static_cast<Matcher*>(matcher)->state->append(Token(token));
        return 0;
    }, -1);
}

CRIT_ENGINE_EXPORT void crit_matcher_close(void* matcher) { delete static_cast<Matcher*>(matcher); }

CRIT_ENGINE_EXPORT const char* crit_constraint_last_error() { return last.c_str(); }
