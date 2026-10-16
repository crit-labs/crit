#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace crit::techniques {
// Accepted-history index. Serial host implementation; no model weights, device
// runtime or context-capacity scans. The caller verifies every proposal and
// appends only after successful accepted-state publication.
class IndexedPromptCopy {
public:
    using Token = std::int32_t;
    struct Limits {
        std::size_t window = 4, history_window = 8, max_draft = 8;
        std::size_t max_matches = 64, max_tokens = 32768;
    };
    explicit IndexedPromptCopy(Limits limits, std::span<const Token> prompt = {}): limits_(limits) {
        if(!limits.window || limits.history_window < limits.window || !limits.max_draft ||
           !limits.max_matches || !limits.max_tokens || limits.window > limits.max_tokens ||
           limits.history_window > limits.max_tokens)
            throw std::invalid_argument("invalid prompt copy bounds");
        append(prompt);
        prompt_end_ = tokens_.size();
    }
    std::span<const Token> tokens() const noexcept { return tokens_; }
    std::size_t indexed_windows() const noexcept { return indexed_; }

    void append(std::span<const Token> accepted) {
        if(accepted.size() > limits_.max_tokens - tokens_.size())
            throw std::length_error("accepted history exceeds capacity");
        if(std::any_of(accepted.begin(), accepted.end(), [](auto t){return t < 0;}))
            throw std::invalid_argument("negative token id");
        // Own aliased input before extending tokens_. Reserve rollback records
        // before mutation; map references survive rehash, unlike iterators.
        const std::vector<Token> input(accepted.begin(), accepted.end());
        std::vector<const Key*> added;
        added.reserve(input.size());
        const auto old = tokens_.size();
        try {
            for(auto token : input) {
                const auto end = tokens_.size();
                tokens_.push_back(token);
                if(end < limits_.window) continue;
                auto [it, inserted] = positions_.try_emplace(key(end));
                try { it->second.push_back(end - limits_.window); }
                catch(...) { if(inserted) positions_.erase(it); throw; }
                added.push_back(&it->first);
            }
        } catch(...) {
            for(auto it = added.rbegin(); it != added.rend(); ++it) {
                auto found = positions_.find(**it);
                found->second.pop_back();
                if(found->second.empty()) positions_.erase(found);
            }
            tokens_.resize(old);
            throw;
        }
        indexed_ += added.size();
    }

    void rewind(std::size_t length) {
        if(length > tokens_.size()) throw std::invalid_argument("rewind extends history");
        // Allocate every key before mutation. Removing only truncated windows
        // restores older occurrences without rebuilding the retained prefix.
        std::vector<Key> removed;
        for(auto end = tokens_.size(); end > length;) {
            --end;
            if(end >= limits_.window) removed.push_back(key(end));
        }
        for(const auto& k : removed) {
            auto it = positions_.find(k);
            it->second.pop_back();
            if(it->second.empty()) positions_.erase(it);
        }
        indexed_ -= removed.size();
        tokens_.resize(length);
        prompt_end_ = std::min(prompt_end_, length);
    }

    std::vector<Token> propose(std::size_t remaining, std::uint32_t vocabulary,
                               std::span<const Token> stops = {}) const {
        if(!vocabulary) throw std::invalid_argument("positive vocabulary required");
        const auto limit = std::min(remaining, limits_.max_draft);
        if(!limit || tokens_.size() <= limits_.window) return {};
        auto it = positions_.find(key(tokens_.size()));
        if(it == positions_.end()) return {};
        std::vector<Token> best;
        std::size_t examined = 0;
        for(auto p = it->second.rbegin(); p != it->second.rend() && examined < limits_.max_matches; ++p, ++examined) {
            const auto end = *p + limits_.window;
            if(*p >= prompt_end_) {
                const auto h = limits_.history_window;
                if(end < h || tokens_.size() < h ||
                   !std::equal(tokens_.begin() + (end-h), tokens_.begin() + end, tokens_.end() - h)) continue;
            }
            const auto count = std::min(limit, tokens_.size() - end);
            std::vector<Token> candidate;
            for(std::size_t j = 0; j < count; ++j) {
                const auto token = tokens_[end+j];
                if(static_cast<std::uint32_t>(token) >= vocabulary) break;
                candidate.push_back(token);
                if(std::find(stops.begin(), stops.end(), token) != stops.end()) break;
            }
            if(candidate.size() > best.size()) best = std::move(candidate);
            if(best.size() == limit) break;
        }
        return best;
    }
private:
    using Key = std::vector<Token>;
    struct Hash {
        std::size_t operator()(const Key& k) const noexcept {
            std::size_t value = 0;
            for(auto token : k) value = value * 16777619U ^ static_cast<std::uint32_t>(token);
            return value;
        }
    };
    Key key(std::size_t end) const {
        return {tokens_.begin() + (end-limits_.window), tokens_.begin() + end};
    }
    Limits limits_;
    std::vector<Token> tokens_;
    // Keep all positions (at most max_tokens) to support exact rewind, while
    // inspecting at most max_matches occurrences on each proposal.
    std::unordered_map<Key, std::vector<std::size_t>, Hash> positions_;
    std::size_t prompt_end_ = 0, indexed_ = 0;
};
}
