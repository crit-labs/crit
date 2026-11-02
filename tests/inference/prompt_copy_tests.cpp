#include "prompt_copy.h"
#include <iostream>
#include <random>

using Index = crit::techniques::IndexedPromptCopy;
using Token = Index::Token;
static void require(bool ok) { if(!ok) throw std::runtime_error("indexed prompt copy mismatch"); }
template<class F> static void rejects(F action) {
    try { action(); } catch(const std::exception&) { return; }
    throw std::runtime_error("expected index rejection");
}
// Independent reverse scan; bounds are held tokens, never configured capacity.
static std::vector<Token> oracle(const std::vector<Token>& h, std::size_t prompt_end,
                                Index::Limits l, std::size_t limit, std::uint32_t vocabulary,
                                std::span<const Token> stops, std::size_t* source=nullptr) {
    limit=std::min(limit,l.max_draft);
    if(!limit || h.size()<=l.window) return {};
    std::vector<Token> best;
    std::size_t matched=0;
    for(std::size_t end=h.size();end>l.window;) {
        --end;
        if(!std::equal(h.begin()+(end-l.window),h.begin()+end,h.end()-l.window)) continue;
        if(matched++==l.max_matches) break;
        if(end-l.window>=prompt_end && (end<l.history_window || h.size()<l.history_window ||
           !std::equal(h.begin()+(end-l.history_window),h.begin()+end,h.end()-l.history_window))) continue;
        std::vector<Token> candidate;
        for(auto i=end;i<h.size() && candidate.size()<limit;++i) {
            if(static_cast<std::uint32_t>(h[i])>=vocabulary) break;
            candidate.push_back(h[i]);
            if(std::find(stops.begin(),stops.end(),h[i])!=stops.end()) break;
        }
        if(candidate.size()>best.size()) { best=candidate; if(source) *source=end; }
        if(best.size()==limit) break;
    }
    return best;
}
// The match behind a draft, counted token by token from the source's end backwards.
static std::size_t oracle_matched(const std::vector<Token>& h, std::size_t source, std::size_t longest) {
    std::size_t k=0;
    while(k<longest && k<source && h[source-1-k]==h[h.size()-1-k]) ++k;
    return k;
}
int main() {
    try {
        std::mt19937 rng(318);
        for(auto capacity:{64U,1000000U}) {
            Index::Limits limits{2,4,6,3,capacity};
            std::vector<Token> history{1,2,3,1,2};
            std::size_t prompt_end=history.size();
            Index index(limits,history);
            require(index.propose(1,10)==std::vector<Token>{3});
            for(int turn=0;turn<500;++turn) {
                if(history.size()>30 || (!history.empty() && rng()%4==0)) {
                    const auto length=rng()%(history.size()+1);
                    history.resize(length); index.rewind(length); prompt_end=std::min(prompt_end,length);
                } else {
                    std::vector<Token> accepted;
                    for(auto count=1+rng()%4;count;--count) accepted.push_back(static_cast<Token>(rng()%4));
                    index.append(accepted); history.insert(history.end(),accepted.begin(),accepted.end());
                }
                require(std::vector<Token>(index.tokens().begin(),index.tokens().end())==history);
                require(index.indexed_windows()==(history.size()>2 ? history.size()-2 : 0));
                const std::vector<Token> stop{2};
                for(auto limit:{0U,1U,5U,9U}) for(auto vocabulary:{3U,10U}) {
                    std::size_t source=0;
                    const auto expected=oracle(history,prompt_end,limits,limit,vocabulary,stop,&source);
                    require(index.propose(limit,vocabulary,stop)==expected);
                    for(std::size_t longest:{2U,5U,64U}) {
                        const auto found=index.propose_matched(limit,vocabulary,stop,longest);
                        require(found.tokens==expected);
                        require(found.matched==(expected.empty() ? 0 : std::max(limits.window,oracle_matched(history,source,longest))));
                    }
                }
            }
            const auto before=std::vector<Token>(index.tokens().begin(),index.tokens().end());
            const std::vector<Token> invalid{-1};
            rejects([&]{index.append(invalid);}); rejects([&]{index.rewind(before.size()+1);});
            require(std::vector<Token>(index.tokens().begin(),index.tokens().end())==before);
        }
        Index bounded({1,1,4,1,4});
        const std::vector<Token> initial{1,2}; bounded.append(initial);
        bounded.append(bounded.tokens()); // Aliased append must own the old view.
        require(bounded.propose(2,5)==initial);
        rejects([&]{bounded.append(initial);});
        bounded.rewind(2); require(bounded.propose(2,5).empty());
        std::cout<<"Indexed accepted history: 1000 mutations and 8000 oracle comparisons passed\n";
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
