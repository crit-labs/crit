#pragma once
#include <crit/core.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

// Optional structured-output service: languages over exact output bytes and request-private token matchers whose
// allowed() set feeds Sampler::sample(logits, AllowedTokens{...}, ...). core/src/constraints/README.md has the rules.
namespace crit::core {
// Most vocabulary entries and total piece bytes one ByteVocabulary holds.
inline constexpr std::size_t constraint_vocabulary_limit = 1048576;
inline constexpr std::size_t constraint_vocabulary_bytes = std::size_t{64} << 20;

// The exact decoded bytes of every token id, as the engine's detokenizer joins them for output (byte-level BPE
// spellings decoded, SentencePiece byte fallback resolved). One explicit EOS has an empty piece; every other empty
// piece marks a token the constraint never allows (special/control tokens). Immutable; share it between matchers.
class ByteVocabulary {
public:
    // invalid_argument for no pieces, more than constraint_vocabulary_limit, or an EOS outside the vocabulary or
    // with a nonempty piece; length_error above constraint_vocabulary_bytes.
    ByteVocabulary(std::vector<std::string> pieces, Token eos);
    std::size_t size() const noexcept { return pieces_.size(); }
    Token eos() const noexcept { return eos_; }
    std::string_view piece(Token) const;                 // out_of_range outside the vocabulary
private:
    std::vector<std::string> pieces_;
    Token eos_ = 0;
};

// Request-private language state over output tokens. size() counts accepted output tokens including EOS; the
// vocabulary stays fixed. append() adds exactly one token or throws without changing state (domain_error for a
// token the language or a limit forbids); rewind(n) restores the exact frontier after n tokens. allowed() returns
// sorted unique ids, only EOS once ended (an absorbing law, so terminal verification rows stay defined), and throws
// domain_error when nothing can continue: it never returns an empty set, so a dead end can never be mistaken for
// "no constraint". fork() copies every mutable frontier. Calls are serialized by the owner and settle before return.
class TokenMatcher {
public:
    virtual ~TokenMatcher() = default;
    virtual std::unique_ptr<TokenMatcher> fork() const = 0;
    virtual std::size_t size() const noexcept = 0;
    virtual std::size_t vocabulary() const noexcept = 0;
    virtual bool ended() const noexcept = 0;
    virtual std::vector<Token> allowed() const = 0;
    virtual void append(Token) = 0;
    virtual void rewind(std::size_t output_tokens) = 0;
};

// A finite set of complete byte strings, independent of how tokens segment them: a token is allowed when its whole
// piece follows the trie; EOS at the end of a complete choice. Bounds: 1..65536 choices, 1 MiB of choice bytes.
class ChoiceGrammar {
public:
    explicit ChoiceGrammar(std::span<const std::string> choices);   // invalid_argument / length_error past bounds
    std::size_t nodes() const noexcept { return terminal_.size(); }
    bool terminal(std::size_t node) const;
    // The node after `bytes` from `node`, or SIZE_MAX when a byte leaves the trie.
    std::size_t walk(std::size_t node, std::string_view bytes) const;
private:
    struct Edge { unsigned char byte = 0; std::uint32_t next = 0; };
    std::vector<std::uint32_t> first_;                   // node -> its edges [first_[n], first_[n + 1]) by byte
    std::vector<Edge> edges_;
    std::vector<bool> terminal_;
};
class ChoiceMatcher final : public TokenMatcher {
public:
    ChoiceMatcher(std::shared_ptr<const ChoiceGrammar>, std::shared_ptr<const ByteVocabulary>);
    std::unique_ptr<TokenMatcher> fork() const override;
    std::size_t size() const noexcept override { return history_.size() - 1; }
    std::size_t vocabulary() const noexcept override;
    bool ended() const noexcept override { return history_.back().ended; }
    std::vector<Token> allowed() const override;
    void append(Token) override;
    void rewind(std::size_t output_tokens) override;
private:
    struct Frontier { std::size_t node = 0; bool ended = false; };
    std::shared_ptr<const ChoiceGrammar> grammar_;
    std::shared_ptr<const ByteVocabulary> vocabulary_;
    std::vector<Frontier> history_{Frontier{}};
    std::size_t next(Token) const;                       // SIZE_MAX when not allowed
};

// One JSON object (OpenAI's json_object mode): strict UTF-8, JSON whitespace around it, nested values within
// `depth`; output bytes and tokens (EOS included) within the other limits. Maxima 256, 1 GiB and 1,048,576.
struct JsonLimits { std::size_t depth = 64, bytes = 1048576, tokens = 32768; };
namespace constraints_detail { struct JsonCursor; }
// Syntax only: duplicate names, escaped surrogate pairing and numeric range are not checked.
class JsonObjectMatcher final : public TokenMatcher {
public:
    explicit JsonObjectMatcher(std::shared_ptr<const ByteVocabulary>, JsonLimits = {});   // invalid_argument
    JsonObjectMatcher(const JsonObjectMatcher&);
    JsonObjectMatcher& operator=(const JsonObjectMatcher&) = delete;
    ~JsonObjectMatcher() override;
    std::unique_ptr<TokenMatcher> fork() const override;
    std::size_t size() const noexcept override;
    std::size_t vocabulary() const noexcept override;
    bool ended() const noexcept override;
    bool complete() const noexcept;                      // a whole object has been read (EOS is allowed)
    std::vector<Token> allowed() const override;
    void append(Token) override;
    void rewind(std::size_t output_tokens) override;
private:
    std::shared_ptr<const ByteVocabulary> vocabulary_;
    JsonLimits limits_;
    std::vector<constraints_detail::JsonCursor> history_;
    bool trial(Token, constraints_detail::JsonCursor& out) const;
};

// A normalized byte context-free grammar: symbols 0..255 are bytes, 256 + i is rule i. The crit_core.constraints
// compiler (JSON schema, byte regex and EBNF subsets) emits it as a "crit-byte-grammar-v1" artifact; engines can
// also build one directly. Bounds: 4,096 rules, 16,384 productions, 262,144 symbols, 4 MiB artifacts. Every rule
// must be defined and productive.
class ByteGrammar {
public:
    struct Production { std::size_t lhs = 0; std::vector<std::size_t> rhs; };
    ByteGrammar(std::size_t rules, std::size_t start, std::vector<Production>);  // invalid_argument
    // Parses an artifact: invalid_argument for a malformed one, length_error above 4 MiB.
    static std::shared_ptr<const ByteGrammar> load(std::string_view artifact);
    const Production& production(std::size_t i) const { return productions_.at(i); }
    const std::vector<std::size_t>& rule(std::size_t i) const { return rules_.at(i); }
    std::size_t rules() const noexcept { return rules_.size(); }
    std::size_t start() const noexcept { return start_; }
    bool nullable(std::size_t rule) const { return nullable_.at(rule); }
private:
    std::size_t start_ = 0;
    std::vector<Production> productions_;
    std::vector<std::vector<std::size_t>> rules_;
    std::vector<bool> nullable_;
};
// Output bytes and tokens (EOS included), Earley states in one byte's chart and in all charts. Maxima 1 MiB,
// 1,048,576, 65,536 and 4,194,304.
struct GrammarLimits { std::size_t bytes = 4096, tokens = 4096, states_per_byte = 4096, total_states = 262144; };
// Bounded Earley recognizer: recursion, ambiguity and nullable rules; a token may cross several productions or split
// a UTF-8 character; EOS only after a complete start rule. Exhausting the chart budget throws length_error and
// leaves the matcher unchanged: states are never pruned and generation never falls back to unconstrained.
class GrammarMatcher final : public TokenMatcher {
public:
    GrammarMatcher(std::shared_ptr<const ByteGrammar>, std::shared_ptr<const ByteVocabulary>, GrammarLimits = {});
    std::unique_ptr<TokenMatcher> fork() const override;
    std::size_t size() const noexcept override { return marks_.size() - 1; }
    std::size_t vocabulary() const noexcept override;
    bool ended() const noexcept override { return marks_.back().ended; }
    bool complete() const noexcept;                      // the output so far is a whole sentence (EOS is allowed)
    std::vector<Token> allowed() const override;
    void append(Token) override;
    void rewind(std::size_t output_tokens) override;
    struct Item { std::size_t production = 0, dot = 0, origin = 0; bool operator==(const Item&) const = default; };
private:
    using Chart = std::vector<Item>;
    struct Mark { std::size_t bytes = 0; bool ended = false; };
    std::shared_ptr<const ByteGrammar> grammar_;
    std::shared_ptr<const ByteVocabulary> vocabulary_;
    GrammarLimits limits_;
    std::vector<Chart> charts_;                          // one per output byte, plus the start
    std::size_t states_ = 0;                             // items in charts_
    std::vector<Mark> marks_{Mark{}};
    // Charts after `piece`, appended to `extension` (empty on entry); false when a byte has no continuation.
    bool scan(std::string_view piece, std::vector<Chart>& extension) const;
    void close(std::vector<Chart>& extension, std::size_t& states) const;
};

// Diagnostics for one raw logits row and a constraint's allowed ids: softmax probabilities normalized within the
// allowed set (before any sampling filter), and that set's mass under the full unfiltered softmax. Logits are finite
// or -inf; ids sorted, unique and in the row; invalid_argument otherwise or when every allowed logit is -inf.
struct ConstraintReadout { std::vector<Token> ids; std::vector<double> probabilities; double original_mass = 0; };
ConstraintReadout constraint_readout(std::span<const float> logits, std::span<const Token> allowed);
}
