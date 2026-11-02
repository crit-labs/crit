// The portable rows (host_nvfp4.h), included once per ISA file with CRIT_PORTABLE_ROWS naming the function: the
// same source compiled for the baseline target and with AVX2 or AVX-512 enabled, so the compiler vectorizes it for
// each. Every inner loop runs over independent lanes in the oracle's order, so every build gives the oracle's bits.
// Included inside namespace crit::techniques::host_nvfp4::detail, in a file built without FP contraction.
namespace {
// Element k of a row: its code through the format's value table.
inline float portable_element(const std::uint8_t* codes, std::size_t k, int bits, const float* values) {
    if(bits == 8) return values[codes[k]];
    const std::uint8_t byte = codes[k / 2];
    return values[k % 2 ? byte >> 4 : byte & 15];
}
// The declared reduction of one token's L lanes.
inline float portable_reduce(float* acc, std::size_t L, Reduce reduce) {
    if(reduce == Reduce::halving) {
        for(std::size_t w = L / 2; w >= 1; w /= 2)
            for(std::size_t i = 0; i < w; ++i) acc[i] = acc[i] + acc[i + w];
        return acc[0];
    }
    float sum = acc[0];
    for(std::size_t i = 1; i < L; ++i) sum = sum + acc[i];
    return sum;
}

// The portable path: the oracle's operations with every inner loop running over independent lanes, so a compiler
// vectorizes it for any target without changing a lane's order. scaled_elements decodes L elements at a time
// (lane j takes element c + j); block_dot runs L blocks at a time (lane j takes block c + j's sequential dot).
// L is a template parameter so every lane loop has a fixed trip count.
template <int L>
void portable_rows_t(const float* values, const Order& order, const MatrixView& m, const float* scales,
                     const float* table, const float* x, std::size_t ldx, int tokens, std::uint32_t mask,
                     std::size_t row0, std::size_t row1, float post, float* out, std::size_t ldo) {
    const std::size_t B = std::size_t(m.format.block), groups = m.cols / B;
    const int bits = m.format.bits;
    const std::size_t code_stride = m.code_stride ? m.code_stride : m.cols * std::size_t(bits) / 8,
                      scale_stride = m.scale_stride ? m.scale_stride : groups;
    alignas(64) float acc[kMaxTokens][L], ws[L];
    thread_local std::vector<float> row_scales;
    row_scales.resize(groups);
    for(std::size_t r = row0; r < row1; ++r) {
        const std::uint8_t* codes = m.codes + r * code_stride;
        const std::uint8_t* sb = m.scales + r * scale_stride;
        float* rs = row_scales.data();
        if(table) for(std::size_t b = 0; b < groups; ++b) rs[b] = table[sb[b]];
        else rs = const_cast<float*>(scales + (r - row0) * groups);
        for(int t = 0; t < tokens; ++t)
            for(int j = 0; j < L; ++j) acc[t][j] = 0.0f;
        if(order.product == Product::block_dot) {
            alignas(64) float d[kMaxTokens][L];
            for(std::size_t c = 0; c < groups; c += L) {
                const std::size_t n = groups - c < std::size_t(L) ? groups - c : std::size_t(L);
                for(int t = 0; t < tokens; ++t)
                    for(int j = 0; j < L; ++j) d[t][j] = 0.0f;
                for(std::size_t e = 0; e < B; ++e) {
                    for(std::size_t j = 0; j < n; ++j) ws[j] = portable_element(codes, (c + j) * B + e, bits, values);
                    for(std::size_t j = n; j < std::size_t(L); ++j) ws[j] = 0.0f;
                    for(int t = 0; t < tokens; ++t) {
                        if(!(mask & (1u << t))) continue;
                        const float* xt = x + std::size_t(t) * ldx;
                        alignas(64) float xv[L];
                        for(std::size_t j = 0; j < n; ++j) xv[j] = xt[(c + j) * B + e];
                        for(std::size_t j = n; j < std::size_t(L); ++j) xv[j] = 0.0f;
                        for(int j = 0; j < L; ++j) d[t][j] = std::fma(ws[j], xv[j], d[t][j]);
                    }
                }
                // lanes past the last block keep their accumulator bits (no fma into them)
                for(int t = 0; t < tokens; ++t)
                    for(std::size_t j = 0; j < n; ++j) acc[t][j] = std::fma(d[t][j], rs[c + j], acc[t][j]);
            }
        } else {
            for(std::size_t c = 0; c < m.cols; c += L) {
                if(bits == 8)
                    for(int j = 0; j < L; ++j) ws[j] = values[codes[c + std::size_t(j)]];
                else
                    for(int j = 0; j < L; j += 2) {
                        const std::uint8_t byte = codes[(c + std::size_t(j)) / 2];
                        ws[j] = values[byte & 15];
                        ws[j + 1] = values[byte >> 4];
                    }
                if(B >= std::size_t(L)) {
                    const float sc = rs[c / B];
                    for(int j = 0; j < L; ++j) ws[j] = ws[j] * sc;
                } else {
                    for(int j = 0; j < L; ++j) ws[j] = ws[j] * rs[(c + std::size_t(j)) / B];
                }
                for(int t = 0; t < tokens; ++t) {
                    if(!(mask & (1u << t))) continue;
                    const float* xt = x + std::size_t(t) * ldx + c;
                    for(int j = 0; j < L; ++j) acc[t][j] = std::fma(ws[j], xt[j], acc[t][j]);
                }
            }
        }
        for(int t = 0; t < tokens; ++t)
            if(mask & (1u << t)) out[std::size_t(t) * ldo + r] = portable_reduce(acc[t], std::size_t(L), order.reduce) * post;
    }
}
}  // namespace

void CRIT_PORTABLE_ROWS(const float* values, const Order& order, const MatrixView& m, const float* scales,
                        const float* table, const float*, const float* x, std::size_t ldx, int tokens,
                        std::uint32_t mask, std::size_t row0, std::size_t row1, float post, float* out,
                        std::size_t ldo) {
    if(order.lanes == 32) portable_rows_t<32>(values, order, m, scales, table, x, ldx, tokens, mask, row0, row1, post, out, ldo);
    else portable_rows_t<16>(values, order, m, scales, table, x, ldx, tokens, mask, row0, row1, post, out, ldo);
}
