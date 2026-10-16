#include "crit/device_graph.h"
#include <algorithm>
#include <limits>
#include <set>

namespace crit::core {
void GraphKey::validate() const {
    if(phase.empty() || representation.empty() || rows.empty() || addresses.empty())
        throw std::invalid_argument("complete graph identity required");
    if(!allocation_generation) throw std::invalid_argument("positive backing allocation generation required");
    std::set<std::pair<std::uint64_t, std::uint64_t>> seen;
    for(const auto& row : rows) {
        if(!row.first || !row.second) throw std::invalid_argument("positive request/generation identities required");
        if(!seen.insert(row).second) throw std::invalid_argument("duplicate row identity");
    }
    if(std::find(addresses.begin(), addresses.end(), std::uintptr_t{0}) != addresses.end())
        throw std::invalid_argument("explicit physical addresses required");
    if(shapes.empty()) throw std::invalid_argument("positive explicit shapes required");
    for(const auto& shape : shapes)
        if(shape.empty() || std::find(shape.begin(), shape.end(), std::size_t{0}) != shape.end())
            throw std::invalid_argument("positive explicit shapes required");
    if(!binding_generations.empty() &&
       (binding_generations.size() != addresses.size() ||
        std::find(binding_generations.begin(), binding_generations.end(), std::uint64_t{0}) != binding_generations.end()))
        throw std::invalid_argument("one positive generation per physical binding required");
}

bool GraphCache::Entry::settled() {
    if(failed_launch) return false;
    // A probe that throws leaves every receipt retained.
    std::vector<std::shared_ptr<GraphReceipt>> live;
    for(const auto& completion : completions)
        if(!completion->done()) live.push_back(completion);
    completions = std::move(live);
    return completions.empty();
}
GraphCache::GraphCache(std::size_t capacity):capacity_(capacity) {
    if(!capacity) throw std::invalid_argument("positive graph capacity required");
}
GraphHandle GraphCache::capture(const GraphKey& key, std::vector<GraphOwner> owners, const Capture& capture_graph) {
    key.validate();
    if(const auto found = entries_.find(key); found != entries_.end()) {
        if(found->second.owners != owners)
            throw std::logic_error("same graph identity cannot refer to new physical owners");
        return found->second.graph;
    }
    if(entries_.size() >= capacity_) throw std::logic_error("explicit settled eviction required");
    if(owners.empty() || std::find(owners.begin(), owners.end(), nullptr) != owners.end())
        throw std::invalid_argument("capture must retain its physical owners");
    if(!capture_graph) throw std::invalid_argument("capture hook required");
    GraphHandle graph;
    try {
        graph = capture_graph(key);
        if(!graph) throw std::logic_error("capture returned no graph");
    } catch(...) {
        // Captured work may exist: release the owners only after backend abort or quiescence proof.
        quarantined_.push_back(std::move(owners));
        throw;
    }
    entries_.emplace(key, Entry{graph, std::move(owners), {}, false});
    return graph;
}
std::shared_ptr<GraphReceipt> GraphCache::replay(const GraphKey& key, const Launch& launch,
                                                 std::span<const std::shared_ptr<GraphReceipt>> prerequisites) {
    key.validate();
    const auto found = entries_.find(key);
    if(found == entries_.end()) throw std::out_of_range("graph identity is not cached");
    auto& entry = found->second;
    if(entry.failed_launch) throw std::logic_error("failed launch requires backend recovery before reuse");
    if(!launch) throw std::invalid_argument("launch hook required");
    for(const auto& receipt : prerequisites)
        if(!receipt) throw std::invalid_argument("prerequisites require settlement and success receipts");
    for(const auto& receipt : prerequisites)
        if(!receipt->done()) throw std::logic_error("graph inputs are not ready for this replay");
    for(const auto& receipt : prerequisites) receipt->result(); // A settled failed upload is not ready content.
    // Reuse can continue without eviction; keep only physical uses still in flight plus the new receipt.
    entry.settled();
    std::shared_ptr<GraphReceipt> completion;
    try {
        completion = launch(entry.graph);
        if(!completion) throw std::logic_error("launch must return a physical completion receipt");
        entry.completions.reserve(entry.completions.size() + 1);
    } catch(...) {
        entry.failed_launch = true; // Submitted work may exist without a receipt.
        throw;
    }
    entry.completions.push_back(completion);
    return completion;
}
void GraphCache::evict(const GraphKey& key, const Destroy& destroy) {
    const auto found = entries_.find(key);
    if(found == entries_.end()) throw std::out_of_range("graph identity is not cached");
    if(!destroy) throw std::invalid_argument("destroy hook required");
    if(!found->second.settled()) throw std::logic_error("graph still accesses retained owners");
    destroy(found->second.graph);
    entries_.erase(found);
}
std::size_t GraphCache::size() const noexcept { return entries_.size(); }
bool GraphCache::contains(const GraphKey& key) const { return entries_.contains(key); }
const GraphCache::Entry& GraphCache::at(const GraphKey& key) const {
    const auto found = entries_.find(key);
    if(found == entries_.end()) throw std::out_of_range("graph identity is not cached");
    return found->second;
}
std::span<const GraphOwner> GraphCache::owners(const GraphKey& key) const { return at(key).owners; }
std::size_t GraphCache::in_flight(const GraphKey& key) const { return at(key).completions.size(); }
bool GraphCache::failed(const GraphKey& key) const { return at(key).failed_launch; }
std::span<const std::vector<GraphOwner>> GraphCache::quarantined() const noexcept { return quarantined_; }

namespace {
std::size_t multiply(std::size_t a, std::size_t b) {
    if(b && a > std::numeric_limits<std::size_t>::max() / b) throw UnsupportedLayout("matrix byte extent overflow");
    return a * b;
}
}
std::size_t element_bytes(MatrixElement element) {
    switch(element) {
    case MatrixElement::f16: case MatrixElement::bf16: return 2;
    case MatrixElement::f32: return 4;
    }
    throw UnsupportedLayout("unsupported matrix element");
}
std::size_t matrix_bytes(const MatrixLayout& layout) {
    if(!layout.rows || !layout.columns || layout.stride < layout.columns)
        throw UnsupportedLayout("positive rows/columns and nonoverlapping row stride required");
    return multiply(multiply(layout.rows, layout.stride), element_bytes(layout.element));
}
StagedGraphPlan::StagedGraphPlan(MatrixLayout input, MatrixLayout output, std::size_t padded_rows,
                                 std::size_t phases, std::size_t bridges)
    :input_(input), output_(output), staged_input_{padded_rows, input.columns, input.columns, input.element},
     staged_output_{padded_rows, output.columns, output.columns, output.element}, phases_(phases) {
    matrix_bytes(input_);
    matrix_bytes(output_);
    if(input.rows != output.rows || padded_rows < input.rows)
        throw UnsupportedLayout("input/output rows must match and fit the padded staging rows");
    if(!phases || phases > 64 || bridges != phases - 1)
        throw UnsupportedLayout("one to 64 phases with one bridge between adjacent phases required");
    matrix_bytes(staged_input_);
    matrix_bytes(staged_output_);
    const auto in_item = element_bytes(input.element), out_item = element_bytes(output.element);
    const auto in_row = input.columns * in_item, out_row = output.columns * out_item;
    input_rows_.reserve(input.rows);
    output_rows_.reserve(output.rows);
    for(std::size_t row = 0; row < input.rows; ++row) {
        input_rows_.push_back({row * in_row, row * input.stride * in_item, in_row});
        output_rows_.push_back({row * output.stride * out_item, row * out_row, out_row});
    }
}
bool StagedGraphPlan::compatible(const MatrixLayout& input, const MatrixLayout& output) const noexcept {
    return input == input_ && output == output_;
}
void StagedGraphPlan::check(const MatrixLayout& input, std::size_t input_bytes, const MatrixLayout& output,
                            std::size_t output_bytes) const {
    if(!compatible(input, output)) throw UnsupportedLayout("capture layout differs");
    if(input_bytes < matrix_bytes(input) || output_bytes < matrix_bytes(output))
        throw UnsupportedLayout("matrix owner does not cover layout");
}
}
