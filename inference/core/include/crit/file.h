#pragma once
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <string>

namespace crit::core {
// Change-detection identity of one file. POSIX: st_dev, st_ino, st_size, st_mtim and st_ctim.
// Windows: FILE_ID_INFO volume serial and 128-bit file id (low 64 bits in file_id, high in
// file_id_high), end-of-file size, FILE_BASIC_INFO LastWriteTime and ChangeTime. Times are
// nanoseconds since the Unix epoch.
struct FileIdentity {
    std::uint64_t device = 0, file_id = 0, size = 0;
    std::int64_t modified_ns = 0, changed_ns = 0;
    std::uint64_t file_id_high = 0;
    friend bool operator==(const FileIdentity&, const FileIdentity&) = default;
};
// alignment: nonzero power of two; chunk_bytes: positive multiple of it. Direct I/O reads aligned
// extents through a second handle verified to be the same file and falls back to buffered reads.
struct ReadOptions {
    std::size_t chunk_bytes = std::size_t(8) << 20, alignment = 4096;
    bool direct = false, overlap = true;
};
struct ReadStats {
    bool direct_used = false;
    std::string fallback_reason = {};   // why direct I/O was not used although requested
    std::uint64_t bytes = 0;            // bytes handed to the consumer
    std::size_t chunks = 0;             // consumer calls
};
// Called on the calling thread, in file order, with nonempty spans that are valid only for the call.
using ChunkConsumer = std::function<void(std::span<const std::byte> bytes, std::uint64_t file_offset)>;

// UTF-8 form of a path for diagnostics; an embedded NUL appears as "\0".
std::string display_path(const std::filesystem::path&);

// A regular file held open read-only. Windows opens with FILE_SHARE_READ, so writers, renames
// and deletes are refused while the handle (or a mapping owner) lives; POSIX cannot prevent
// mutation, so verify_unchanged() detects it. OS failures throw std::system_error carrying
// errno/GetLastError with "<path>: <operation>"; non-regular files throw std::invalid_argument.
// Move-only; a moved-from object throws std::logic_error from every throwing member.
class ReadOnlyFile {
public:
    ReadOnlyFile(ReadOnlyFile&&) noexcept;
    ReadOnlyFile& operator=(ReadOnlyFile&&) noexcept;
    ReadOnlyFile(const ReadOnlyFile&) = delete;
    ReadOnlyFile& operator=(const ReadOnlyFile&) = delete;
    ~ReadOnlyFile();
    // follow_symlinks = false refuses a symbolic link (or reparse point) as the final component. A
    // relative path is made absolute once here; path() stays as given. std::invalid_argument for a
    // path containing a NUL character.
    static ReadOnlyFile open(const std::filesystem::path& path, bool follow_symlinks = true);
    const std::filesystem::path& path() const noexcept;
    const FileIdentity& identity() const noexcept;    // captured on the handle at open
    FileIdentity handle_identity() const;
    FileIdentity path_identity() const;
    // std::runtime_error("<path>: file changed since open ...") unless the handle and the path
    // both still match identity(); a path that no longer resolves counts as changed.
    void verify_unchanged() const;
    // Exactly dst.size() bytes at offset from the held handle. std::out_of_range past identity().size;
    // std::runtime_error "unexpected end of file" when the file shrank.
    void read_exact(std::uint64_t offset, std::span<std::byte> dst) const;
    // read_exact through reader slot `worker`, for several threads reading one file at once. Slot 0, and any id at or
    // past readers(), uses the held handle. On Windows synchronous reads through one handle are serialised by its file
    // object, so slots 1 .. readers() - 1 each own a private handle reopened from the held one (same file, random-
    // access hint); POSIX pread is concurrent on one descriptor, so every slot uses it there.
    void read_exact(std::uint64_t offset, std::span<std::byte> dst, std::size_t worker) const;
    // Sets the reader slots (at least 1) and opens their private handles, closing the previous ones. Call it before
    // concurrent reads begin, not during them: threads already running must be ordered after it (for example by
    // the mutex or queue that hands them work). std::system_error when a handle cannot be reopened, and the
    // allocation's error for a count that cannot be held; no slot changes then. std::logic_error on a moved-from
    // file.
    void set_readers(std::size_t readers);
    std::size_t readers() const noexcept;           // 0 on a moved-from file
    std::size_t private_readers() const noexcept;   // private handles held (0 on POSIX and on a moved-from file)
    // [offset, offset + length) in chunks; with overlap one reader thread fills the next chunk while
    // the consumer runs. A consumer exception propagates after the in-flight read finishes. At most
    // two chunk buffers are allocated. Invalid options: std::invalid_argument; past end: std::out_of_range.
    ReadStats read_chunks(std::uint64_t offset, std::uint64_t length, const ChunkConsumer& consume,
                          const ReadOptions& options = {}) const;
    // Whole-file read-only view, created once on first use (thread-safe); empty for an empty file.
    // Verifies the handle identity before mapping. POSIX truncation underneath raises SIGBUS.
    std::span<const std::byte> map() const;
    // Keeps the mapping (and on Windows a deny-write handle) alive past this object.
    std::shared_ptr<const void> mapping_owner() const;
    void advise_willneed(std::uint64_t offset, std::uint64_t length) const noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    explicit ReadOnlyFile(std::unique_ptr<Impl>);
    const Impl& live() const;
};
}
