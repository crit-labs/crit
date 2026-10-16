#pragma once
#include <crit/file.h>
#include <crit/json.h>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace crit::core {
// Storage dtypes of safetensors files. Values are never decoded here; only their bit widths matter.
enum class DType : std::uint8_t { boolean, u8, i8, u16, i16, u32, i32, u64, i64,
                                  f8_e4m3, f8_e5m2, f8_e8m0, f16, bf16, f32, f64, f4, f6_e2m3, f6_e3m2 };
std::string_view dtype_name(DType dtype) noexcept;            // stored spelling: "BOOL", "U8", ..., "F8_E8M0", "F4"; "" when invalid
std::uint32_t dtype_bits(DType dtype) noexcept;               // 8..64, F4 = 4, F6_* = 6; 0 when invalid
std::optional<DType> parse_dtype(std::string_view name) noexcept;   // exact, case-sensitive stored spelling

// One tensor record. begin/end are absolute file offsets (8 + header length + data_offsets).
struct TensorInfo {
    std::string name = {};
    DType dtype = DType::boolean;
    std::vector<std::uint64_t> shape = {};
    std::uint32_t shard = 0;                    // index into Checkpoint::shards()
    std::uint64_t begin = 0, end = 0;
    std::uint64_t nbytes() const noexcept;      // end - begin (0 when end < begin)
    std::uint64_t elements() const noexcept;    // product of shape, 1 for a scalar; saturates at UINT64_MAX
    bool element_aligned() const noexcept;      // begin % max(1, bits / 8) == 0 (always true below one byte)
};

// Bounds checked before memory is reserved. Every limit must be in [1, 2^63 - 1] (std::invalid_argument).
struct SafetensorsLimits {
    std::uint64_t header_bytes = 100'000'000, index_bytes = std::uint64_t(64) << 20, name_bytes = 4096,
                  dimension = std::uint64_t(1) << 48;
    std::uint32_t tensors = 1u << 20, shards = 4096, rank = 16;
    std::uint64_t file_bytes = std::uint64_t(1) << 42;          // one file; also bounds header_only offsets
    // The index plus every shard and extra file, each by its size or, with header_only, by its last tensor's end
    // when that is larger.
    std::uint64_t total_bytes = std::uint64_t(1) << 46;
    // Memory kept after open, charged before it is reserved: per file the header and __metadata__ bytes plus
    // sizeof(TensorInfo) + name bytes + 8 x rank per tensor; a Checkpoint adds 4 per tensor and sums its files.
    std::uint64_t retained_bytes = std::uint64_t(1) << 30;
};
// strict: index keys limited to metadata/weight_map, metadata to total_size, which must match the headers.
// compatible: also metadata.total_parameters; unknown keys (one note per group, naming at most eight) and a
// stale total_size become notes().
enum class IndexPolicy : std::uint8_t { strict, compatible };
struct CheckpointOptions {
    SafetensorsLimits limits = {};
    IndexPolicy index = IndexPolicy::compatible;
    bool follow_symlinks = true;     // false refuses a symbolic link as any opened file (the audit policy)
    bool header_only = false;        // skip payload bounds and coverage to the end; byte access throws logic_error
    std::vector<std::filesystem::path> extra_files = {};   // flat file names beside the source, e.g. a draft-layer file
};

// One validated .safetensors file held open read-only (README: validation order). Malformed data throws
// FormatError naming the file; OS failures std::system_error; bad limits std::invalid_argument. Move-only.
class SafetensorsFile {
public:
    static SafetensorsFile open(const std::filesystem::path& path, const CheckpointOptions& options = {});
    SafetensorsFile(SafetensorsFile&& other) noexcept;
    SafetensorsFile& operator=(SafetensorsFile&& other) noexcept;
    const ReadOnlyFile& file() const noexcept { return file_; }
    std::uint64_t header_length() const noexcept { return header_.size(); }        // N after the 8-byte prefix
    std::uint64_t payload_offset() const noexcept { return 8 + header_length(); }
    std::string_view header_bytes() const noexcept { return header_; }
    std::span<const std::pair<std::string, std::string>> metadata() const noexcept { return metadata_; }   // __metadata__
    // Header order. Inside a Checkpoint this is the shard's slice of Checkpoint::tensors(), not a copy.
    std::span<const TensorInfo> tensors() const noexcept { return tensors_; }
    bool header_only() const noexcept { return header_only_; }
    std::uint64_t retained_bytes() const noexcept { return retained_; }           // charged to limits.retained_bytes
private:
    friend class Checkpoint;
    ReadOnlyFile file_;
    std::string header_;
    std::vector<std::pair<std::string, std::string>> metadata_;
    std::vector<TensorInfo> owned_;              // the records of a standalone file; empty inside a Checkpoint
    std::span<const TensorInfo> tensors_;
    std::uint64_t retained_ = 0;
    bool header_only_ = false;
    explicit SafetensorsFile(ReadOnlyFile file);
    // retained_before: what the checkpoint already charged to limits.retained_bytes.
    static SafetensorsFile open_shard(const std::filesystem::path& path, const CheckpointOptions& options, std::uint32_t shard,
                                      std::uint64_t retained_before);
};

// A .safetensors file, a flat-shard *.safetensors.index.json or a directory resolving to one, with every header
// cross-checked against the index. Const members are safe to call from several threads. Move-only.
class Checkpoint {
public:
    static Checkpoint open(const std::filesystem::path& path, const CheckpointOptions& options = {});
    const std::filesystem::path& source() const noexcept { return source_; }      // the index or file opened
    std::span<const SafetensorsFile> shards() const noexcept { return shards_; }  // index shards by name, then extras
    // Every tensor once, shard by shard in shards() order and in header order within a shard.
    std::span<const TensorInfo> tensors() const noexcept { return tensors_; }
    const TensorInfo* find(std::string_view name) const noexcept;                 // binary search of a name index
    const TensorInfo& at(std::string_view name) const;                             // std::out_of_range when absent
    std::span<const std::string> notes() const noexcept { return notes_; }        // compatible-policy findings
    const ReadOnlyFile* index_file() const noexcept { return index_ ? &*index_ : nullptr; }
    bool header_only() const noexcept { return header_only_; }
    std::uint64_t retained_bytes() const noexcept { return retained_; }           // every file's charge + 4 per tensor

    // The tensor must be one of tensors() (std::invalid_argument otherwise); header_only throws std::logic_error.
    struct View { std::span<const std::byte> bytes = {}; std::shared_ptr<const void> owner = {}; };
    View map(const TensorInfo& tensor) const;     // zero-copy read-only view; owner keeps it valid past *this
    void read(const TensorInfo& tensor, std::span<std::byte> dst) const;   // dst.size() == nbytes() else invalid_argument
    // Consumer offsets are absolute file offsets (subtract tensor.begin for an offset in the tensor).
    ReadStats read_chunks(const TensorInfo& tensor, const ChunkConsumer& consume, const ReadOptions& options = {}) const;
    // One pass over a shard in file order handing each tensor's bytes to consume (zero-byte tensors get no call),
    // then verify_unchanged(). With sha256_hex it reads the whole file and stores its digest once all succeeded.
    using TensorConsumer = std::function<void(const TensorInfo& tensor, std::span<const std::byte> bytes,
                                              std::uint64_t offset_in_tensor)>;
    ReadStats stream_shard(std::uint32_t shard, const TensorConsumer& consume, const ReadOptions& options = {},
                           std::string* sha256_hex = nullptr) const;
    void verify_unchanged() const;                // the index and every shard

    // A crit run-record manifest row: path relative to the source directory with '/', size, lowercase hex.
    struct FileDigest {
        std::string path = {};
        std::uint64_t bytes = 0;
        std::string sha256 = {};
        friend bool operator==(const FileDigest&, const FileDigest&) = default;
    };
    std::vector<FileDigest> digest_files(const ReadOptions& options = {.direct = true}) const;   // sorted by path
    // Every checkpoint file needs exactly one row with equal bytes and sha256 (rows for other files are ignored).
    // FormatError naming the file on absence or a size or hash mismatch; all sizes are checked before hashing.
    void verify_digests(std::span<const FileDigest> expected, const ReadOptions& options = {.direct = true}) const;
private:
    std::filesystem::path source_;
    std::optional<ReadOnlyFile> index_;
    std::vector<SafetensorsFile> shards_;
    std::vector<TensorInfo> tensors_;            // owns every record; shards_ hold slices of it
    std::vector<std::uint32_t> by_name_;         // positions in tensors_ sorted by name (bytewise)
    std::vector<std::string> notes_;
    std::uint64_t retained_ = 0;
    bool header_only_ = false;
    Checkpoint() = default;
    const TensorInfo& owned(const TensorInfo& tensor) const;
};
}
