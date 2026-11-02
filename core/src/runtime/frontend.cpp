// Text frontend services: Engine's frontend access, decoders that own their frontend, and request text streams.
#include "engine.h"
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <typeinfo>
#include <utility>
#include <vector>

namespace crit::core {
namespace detail {
namespace {
class OwnedTextDecoder final : public TextDecoder {
    // Destroy the decoder before the frontend whose tables it may borrow.
    std::shared_ptr<const TextFrontend> owner_;
    std::unique_ptr<TextDecoder> decoder_;
    std::uint32_t vocabulary_;
    bool closed_ = false;
public:
    OwnedTextDecoder(std::shared_ptr<const TextFrontend> owner,
                     std::unique_ptr<TextDecoder> decoder, std::uint32_t vocabulary)
        : owner_(std::move(owner)), decoder_(std::move(decoder)), vocabulary_(vocabulary) {}
    std::string push(std::span<const Token> tokens) override {
        if(closed_) throw std::logic_error("text decoder is closed");
        validate_tokens(tokens,vocabulary_);
        try { return decoder_->push(tokens); }
        catch(...) { closed_ = true; throw; }
    }
    std::string finish() override {
        if(closed_) throw std::logic_error("text decoder is closed");
        closed_ = true;
        return decoder_->finish();
    }
};
}
std::unique_ptr<TextStream> make_text_stream(const std::shared_ptr<const TextFrontend>& frontend, std::span<const Token> prompt,
                                             const std::vector<std::string>& stops, std::uint32_t vocabulary) {
    std::unique_ptr<TextDecoder> decoder;
    try { decoder = frontend->create_decoder(); }
    catch(const std::logic_error& e) {
        if(typeid(e) != typeid(std::logic_error)) throw;
        decoder = offset_decoder(frontend,prompt);
    }
    if(!decoder) throw std::runtime_error("text frontend returned null decoder");
    return std::make_unique<TextStream>(std::make_unique<OwnedTextDecoder>(frontend,std::move(decoder),vocabulary),stops);
}
}
std::unique_ptr<TextDecoder> TextFrontend::create_decoder() const {
    throw std::logic_error("text frontend lacks incremental decoding");
}
std::vector<Token> Engine::tokenize(std::string_view text) {
    if(!impl_) throw std::logic_error("engine is moved from");
    if(impl_->frontend) {
        auto tokens = impl_->frontend->tokenize(text);
        detail::validate_tokens(tokens,impl_->declaration.vocabulary);
        return tokens;
    }
    std::lock_guard lock(impl_->wakeup->mutex);
    if(impl_->busy || impl_->poisoned) throw std::runtime_error("engine unavailable for frontend access");
    auto tokens = impl_->model->tokenize(text);
    detail::validate_tokens(tokens,impl_->declaration.vocabulary);
    return tokens;
}
std::string Engine::detokenize(std::span<const Token> tokens) {
    if(!impl_) throw std::logic_error("engine is moved from");
    detail::validate_tokens(tokens,impl_->declaration.vocabulary);
    if(impl_->frontend) return impl_->frontend->detokenize(tokens);
    std::lock_guard lock(impl_->wakeup->mutex);
    if(impl_->busy || impl_->poisoned) throw std::runtime_error("engine unavailable for frontend access");
    return impl_->model->detokenize(tokens);
}
std::unique_ptr<TextDecoder> Engine::create_decoder() {
    if(!impl_) throw std::logic_error("engine is moved from");
    auto frontend = impl_->frontend;
    if(!frontend) throw std::logic_error("model lacks an independent text frontend");
    auto decoder = frontend->create_decoder();
    if(!decoder) throw std::runtime_error("text frontend returned null decoder");
    return std::make_unique<detail::OwnedTextDecoder>(std::move(frontend),std::move(decoder),impl_->declaration.vocabulary);
}
}
