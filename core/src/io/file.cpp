#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE   // O_DIRECT
#endif
#include "crit/file.h"
#include <algorithm>
#include <cerrno>
#include <condition_variable>
#include <cstddef>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <new>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <utility>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace crit::core {
namespace {
namespace fs = std::filesystem;
#ifdef _WIN32
using Native = HANDLE;
std::error_code last_error() { return {int(GetLastError()), std::system_category()}; }
#else
using Native = int;
std::error_code last_error() { return {errno, std::generic_category()}; }
#endif
const fs::path empty_path;
const FileIdentity empty_identity;

[[noreturn]] void fail_os(const std::string& display, const char* operation, std::error_code code) {
    throw std::system_error(code, display + ": " + operation);
}
// Owns one OS handle or descriptor.
class Handle {
public:
    Handle() = default;
    explicit Handle(Native native) : native_(native) {}
    Handle(Handle&& other) noexcept : native_(std::exchange(other.native_, invalid())) {}
    Handle& operator=(Handle&& other) noexcept {
        if(this != &other) { reset(); native_ = std::exchange(other.native_, invalid()); }
        return *this;
    }
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
    ~Handle() { reset(); }
    Native get() const noexcept { return native_; }
    explicit operator bool() const noexcept { return native_ != invalid(); }
    void reset() noexcept {
#ifdef _WIN32
        if(*this) CloseHandle(native_);
#else
        if(*this) ::close(native_);
#endif
        native_ = invalid();
    }
#ifdef _WIN32
    static Native invalid() noexcept { return INVALID_HANDLE_VALUE; }
#else
    static Native invalid() noexcept { return -1; }
#endif
private:
    Native native_ = invalid();
};
// Uninitialized storage aligned for direct I/O.
class AlignedBuffer {
public:
    AlignedBuffer() = default;
    AlignedBuffer(std::size_t bytes, std::size_t alignment)
        : alignment_(std::max(alignment, alignof(std::max_align_t))),
          data_(static_cast<std::byte*>(::operator new(bytes, std::align_val_t(alignment_)))) {}
    AlignedBuffer(AlignedBuffer&& other) noexcept
        : alignment_(other.alignment_), data_(std::exchange(other.data_, nullptr)) {}
    AlignedBuffer& operator=(AlignedBuffer&& other) noexcept {
        if(this != &other) {
            release();
            alignment_ = other.alignment_;
            data_ = std::exchange(other.data_, nullptr);
        }
        return *this;
    }
    ~AlignedBuffer() { release(); }
    std::byte* get() const noexcept { return data_; }
private:
    std::size_t alignment_ = 0;
    std::byte* data_ = nullptr;
    void release() noexcept {
        if(data_) ::operator delete(data_, std::align_val_t(alignment_));
        data_ = nullptr;
    }
};
struct ReadResult { std::size_t bytes = 0; std::error_code error; };

#ifdef _WIN32
// FILETIME ticks (100 ns since 1601) to nanoseconds since the Unix epoch.
std::int64_t filetime_ns(LONGLONG ticks) {
    constexpr LONGLONG epoch = 116444736000000000LL;
    const LONGLONG unix_ticks = ticks - epoch;
    if(unix_ticks > std::numeric_limits<std::int64_t>::max() / 100) return std::numeric_limits<std::int64_t>::max();
    if(unix_ticks < std::numeric_limits<std::int64_t>::min() / 100) return std::numeric_limits<std::int64_t>::min();
    return std::int64_t(unix_ticks) * 100;
}
FileIdentity identity_of(HANDLE handle, const std::string& display) {
    FileIdentity id;
    FILE_ID_INFO ids{};
    if(GetFileInformationByHandleEx(handle, FileIdInfo, &ids, DWORD(sizeof ids))) {
        id.device = ids.VolumeSerialNumber;
        std::memcpy(&id.file_id, ids.FileId.Identifier, sizeof id.file_id);
        std::memcpy(&id.file_id_high, ids.FileId.Identifier + sizeof id.file_id, sizeof id.file_id_high);
    } else {
        BY_HANDLE_FILE_INFORMATION info{};
        if(!GetFileInformationByHandle(handle, &info)) fail_os(display, "stat", last_error());
        id.device = info.dwVolumeSerialNumber;
        id.file_id = (std::uint64_t(info.nFileIndexHigh) << 32) | info.nFileIndexLow;
    }
    FILE_STANDARD_INFO standard{};
    FILE_BASIC_INFO basic{};
    if(!GetFileInformationByHandleEx(handle, FileStandardInfo, &standard, DWORD(sizeof standard)) ||
       !GetFileInformationByHandleEx(handle, FileBasicInfo, &basic, DWORD(sizeof basic))) fail_os(display, "stat", last_error());
    id.size = std::uint64_t(standard.EndOfFile.QuadPart);
    id.modified_ns = filetime_ns(basic.LastWriteTime.QuadPart);
    id.changed_ns = filetime_ns(basic.ChangeTime.QuadPart);
    return id;
}
ReadResult read_at(HANDLE handle, std::byte* dst, std::size_t bytes, std::uint64_t offset, bool direct) {
    ReadResult result;
    while(result.bytes < bytes) {
        const DWORD want = DWORD(std::min<std::size_t>(bytes - result.bytes, std::size_t(1) << 30));
        const std::uint64_t at = offset + result.bytes;
        OVERLAPPED position{};
        position.Offset = DWORD(at & 0xffffffffu);
        position.OffsetHigh = DWORD(at >> 32);
        DWORD got = 0;
        if(!ReadFile(handle, dst + result.bytes, want, &got, &position)) {
            const DWORD error = GetLastError();
            if(error != ERROR_HANDLE_EOF) result.error = std::error_code(int(error), std::system_category());
            break;
        }
        if(got == 0) break;
        result.bytes += got;
        if(direct && got < want) break;   // end of file: never continue at an unaligned offset
    }
    return result;
}
#else
std::int64_t saturating_ns(std::int64_t seconds, std::int64_t nanoseconds) {
    constexpr std::int64_t giga = 1000000000;
    constexpr auto max = std::numeric_limits<std::int64_t>::max(), min = std::numeric_limits<std::int64_t>::min();
    if(seconds > max / giga - 1) return max;
    if(seconds < min / giga + 1) return min;
    return seconds * giga + nanoseconds;
}
FileIdentity identity_of(const struct stat& status) {
    FileIdentity id;
    id.device = std::uint64_t(status.st_dev);
    id.file_id = std::uint64_t(status.st_ino);
    id.size = status.st_size > 0 ? std::uint64_t(status.st_size) : 0;
#ifdef __APPLE__
    id.modified_ns = saturating_ns(std::int64_t(status.st_mtimespec.tv_sec), status.st_mtimespec.tv_nsec);
    id.changed_ns = saturating_ns(std::int64_t(status.st_ctimespec.tv_sec), status.st_ctimespec.tv_nsec);
#else
    id.modified_ns = saturating_ns(std::int64_t(status.st_mtim.tv_sec), status.st_mtim.tv_nsec);
    id.changed_ns = saturating_ns(std::int64_t(status.st_ctim.tv_sec), status.st_ctim.tv_nsec);
#endif
    return id;
}
int open_retry(const char* path, int flags) {
    int fd;
    do fd = ::open(path, flags); while(fd < 0 && errno == EINTR);
    return fd;
}
ReadResult read_at(int fd, std::byte* dst, std::size_t bytes, std::uint64_t offset, bool direct) {
    ReadResult result;
    while(result.bytes < bytes) {
        const std::size_t want = std::min<std::size_t>(bytes - result.bytes, std::size_t(1) << 30);
        const std::uint64_t at = offset + result.bytes;
        if(at > std::uint64_t(std::numeric_limits<off_t>::max())) { result.error = std::make_error_code(std::errc::value_too_large); break; }
        const ssize_t got = ::pread(fd, dst + result.bytes, want, off_t(at));
        if(got < 0) {
            if(errno == EINTR) continue;
            result.error = last_error();
            break;
        }
        if(got == 0) break;
        result.bytes += std::size_t(got);
        if(direct && std::size_t(got) < want) break;   // end of file: never continue at an unaligned offset
    }
    return result;
}
#endif

// The lazily created whole-file view; mapping_owner() hands out shared ownership of it.
struct Mapping {
    const std::byte* data = nullptr;
    std::size_t bytes = 0;
#ifdef _WIN32
    HANDLE section = nullptr;
    Handle file;   // duplicate of the deny-write handle, so writers stay refused while the view lives
#endif
    Mapping() = default;
    Mapping(const Mapping&) = delete;
    Mapping& operator=(const Mapping&) = delete;
    ~Mapping() {
#ifdef _WIN32
        if(data) UnmapViewOfFile(data);
        if(section) CloseHandle(section);
#else
        if(data) munmap(const_cast<std::byte*>(data), bytes);
#endif
    }
};
std::string differences(const FileIdentity& a, const FileIdentity& b) {
    std::string out;
    auto add = [&](bool differ, const char* name) { if(differ) out += (out.empty() ? "" : ", ") + std::string(name); };
    add(a.device != b.device || a.file_id != b.file_id || a.file_id_high != b.file_id_high, "file id");
    add(a.size != b.size, "size");
    add(a.modified_ns != b.modified_ns, "modified time");
    add(a.changed_ns != b.changed_ns, "change time");
    return out;
}
std::uint64_t round_up(std::uint64_t value, std::uint64_t alignment) { return (value + alignment - 1) & ~(alignment - 1); }
}

std::string display_path(const std::filesystem::path& path) {
    try {
        const auto text = path.u8string();
        std::string out;
        out.reserve(text.size());
        for(const auto c:text) {
            if(c == 0) out += "\\0";   // what() is a C string; an embedded NUL would cut the message
            else out += char(c);
        }
        return out;
    } catch(...) { return "<unrepresentable path>"; }
}

struct ReadOnlyFile::Impl {
    fs::path path;       // as given
    fs::path resolved;   // absolute form used for every OS call, so later cwd changes do not matter
    std::string display;
    FileIdentity identity;
    bool follow = true;
    Handle handle;
    mutable std::mutex mapping_mutex;
    mutable std::shared_ptr<const Mapping> mapping;

    FileIdentity handle_identity() const {
#ifdef _WIN32
        return identity_of(handle.get(), display);
#else
        struct stat status{};
        if(::fstat(handle.get(), &status) != 0) fail_os(display, "stat", last_error());
        return identity_of(status);
#endif
    }
    // A second handle to the same file for direct reads, or an empty handle and the reason.
    Handle open_direct(std::string& why) const {
#if defined(_WIN32)
        const HANDLE reopened = ReOpenFile(handle.get(), GENERIC_READ, FILE_SHARE_READ, FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN);
        if(reopened == INVALID_HANDLE_VALUE) { why = "direct open refused: " + last_error().message(); return {}; }
        return Handle(reopened);
#elif defined(__linux__) || defined(__APPLE__)
#if defined(__linux__)
        const int flags = O_RDONLY | O_CLOEXEC | O_NOCTTY | O_NONBLOCK | O_DIRECT;
        // Reopen the held descriptor itself when /proc is available (no path race), else the path.
        const std::string held = "/proc/self/fd/" + std::to_string(handle.get());
        int fd = open_retry(held.c_str(), flags);
        if(fd < 0) fd = open_retry(resolved.c_str(), flags | (follow ? 0 : O_NOFOLLOW));
#else
        int fd = open_retry(resolved.c_str(), O_RDONLY | O_CLOEXEC | O_NOCTTY | O_NONBLOCK | (follow ? 0 : O_NOFOLLOW));
#endif
        if(fd < 0) { why = "direct open refused: " + last_error().message(); return {}; }
        Handle second(fd);
        struct stat status{};
        if(::fstat(fd, &status) != 0) { why = "direct open refused: " + last_error().message(); return {}; }
        if(std::uint64_t(status.st_dev) != identity.device || std::uint64_t(status.st_ino) != identity.file_id) {
            why = "direct open refused: the path now names a different file";
            return {};
        }
#if defined(__APPLE__)
        if(::fcntl(fd, F_NOCACHE, 1) == -1) { why = "direct open refused: F_NOCACHE: " + last_error().message(); return {}; }
#endif
        return second;
#else
        why = "direct I/O is not supported on this platform";
        return {};
#endif
    }
    std::shared_ptr<const Mapping> mapped() const {
        std::lock_guard lock(mapping_mutex);
        if(mapping) return mapping;
        const auto now = handle_identity();
        if(now != identity) throw std::runtime_error(display + ": file changed since open (" + differences(identity, now) + ")");
        if(identity.size > std::numeric_limits<std::size_t>::max()) throw std::runtime_error(display + ": file too large to map");
        auto view = std::make_shared<Mapping>();
        view->bytes = std::size_t(identity.size);
#ifdef _WIN32
        HANDLE duplicate = nullptr;
        if(!DuplicateHandle(GetCurrentProcess(), handle.get(), GetCurrentProcess(), &duplicate, 0, FALSE, DUPLICATE_SAME_ACCESS))
            fail_os(display, "map", last_error());
        view->file = Handle(duplicate);
        if(view->bytes) {
            view->section = CreateFileMappingW(handle.get(), nullptr, PAGE_READONLY, 0, 0, nullptr);
            if(!view->section) fail_os(display, "map", last_error());
            view->data = static_cast<const std::byte*>(MapViewOfFile(view->section, FILE_MAP_READ, 0, 0, 0));
            if(!view->data) fail_os(display, "map", last_error());
        }
#else
        if(view->bytes) {
            void* pointer = ::mmap(nullptr, view->bytes, PROT_READ, MAP_SHARED, handle.get(), 0);
            if(pointer == MAP_FAILED) fail_os(display, "map", last_error());
            view->data = static_cast<const std::byte*>(pointer);
        }
#endif
        mapping = view;
        return view;
    }
    // Streams one pass over the region. Returns false only when the first direct read fails before
    // the consumer saw any byte (the caller then retries buffered); every other failure throws.
    bool stream(Native source, bool direct, std::uint64_t offset, std::uint64_t length, const ChunkConsumer& consume,
                const ReadOptions& options, ReadStats& stats, std::error_code& refused) const {
        const std::uint64_t alignment = options.alignment, end = offset + length;
        const std::uint64_t base = direct ? offset & ~(alignment - 1) : offset;
        const std::uint64_t read_end = direct ? round_up(end, alignment) : end;
        const std::uint64_t chunk = std::min<std::uint64_t>(options.chunk_bytes, read_end - base);
        const std::uint64_t chunks = (end - base + chunk - 1) / chunk;
        const bool overlap = options.overlap && chunks > 1;
        AlignedBuffer buffers[2];
        buffers[0] = AlignedBuffer(std::size_t(chunk), options.alignment);
        if(overlap) buffers[1] = AlignedBuffer(std::size_t(chunk), options.alignment);
        auto buffer = [&](std::uint64_t i) { return buffers[overlap ? i % 2 : 0].get(); };
        auto read_chunk = [&](std::uint64_t i) {
            const std::uint64_t at = base + i * chunk;
            return read_at(source, buffer(i), std::size_t(std::min(chunk, read_end - at)), at, direct);
        };
        // false: first direct read refused; throws on any other failure or a short read.
        auto check = [&](std::uint64_t i, const ReadResult& got) {
            const std::uint64_t at = base + i * chunk;
            if(got.error) {
                if(direct && i == 0 && got.bytes == 0) { refused = got.error; return false; }
                fail_os(display, "read", got.error);
            }
            if(got.bytes < std::min(chunk, end - at))
                throw std::runtime_error(display + ": unexpected end of file at offset " + std::to_string(at + got.bytes));
            return true;
        };
        auto deliver = [&](std::uint64_t i) {
            const std::uint64_t at = base + i * chunk;
            const std::uint64_t lo = std::max(at, offset), hi = std::min(at + chunk, end);
            consume(std::span<const std::byte>(buffer(i) + (lo - at), std::size_t(hi - lo)), lo);
            stats.bytes += hi - lo;
            ++stats.chunks;
        };
        if(!overlap) {
            for(std::uint64_t i = 0; i < chunks; ++i) {
                if(!check(i, read_chunk(i))) return false;
                deliver(i);
            }
            return true;
        }
        // One reader thread fills slot i % 2 once chunk i - 2 is released; the consumer runs here.
        struct Shared {
            std::mutex mutex;
            std::condition_variable changed;
            ReadResult results[2];
            bool ready[2] = {false, false};
            std::uint64_t released = 0;
            bool stop = false;
        } shared;
        struct Reader {   // joins on every exit, so no read outlives the buffers
            Shared& shared;
            std::thread thread;
            ~Reader() {
                if(!thread.joinable()) return;
                { std::lock_guard lock(shared.mutex); shared.stop = true; }
                shared.changed.notify_all();
                thread.join();
            }
        } reader{shared, {}};
        reader.thread = std::thread([&] {
            for(std::uint64_t i = 0; i < chunks; ++i) {
                {
                    std::unique_lock lock(shared.mutex);
                    shared.changed.wait(lock, [&] { return shared.stop || i < shared.released + 2; });
                    if(shared.stop) return;
                }
                const ReadResult got = read_chunk(i);
                const std::uint64_t at = base + i * chunk;
                {
                    std::lock_guard lock(shared.mutex);
                    shared.results[i % 2] = got;
                    shared.ready[i % 2] = true;
                }
                shared.changed.notify_all();
                if(got.error || got.bytes < std::min(chunk, end - at)) return;   // the consumer side reports it
            }
        });
        for(std::uint64_t i = 0; i < chunks; ++i) {
            ReadResult got;
            {
                std::unique_lock lock(shared.mutex);
                shared.changed.wait(lock, [&] { return shared.ready[i % 2]; });
                got = shared.results[i % 2];
                shared.ready[i % 2] = false;
            }
            if(!check(i, got)) return false;
            deliver(i);
            { std::lock_guard lock(shared.mutex); shared.released = i + 1; }
            shared.changed.notify_all();
        }
        return true;
    }
};

ReadOnlyFile::ReadOnlyFile(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
ReadOnlyFile::ReadOnlyFile(ReadOnlyFile&&) noexcept = default;
ReadOnlyFile& ReadOnlyFile::operator=(ReadOnlyFile&&) noexcept = default;
ReadOnlyFile::~ReadOnlyFile() = default;
const ReadOnlyFile::Impl& ReadOnlyFile::live() const {
    if(!impl_) throw std::logic_error("read-only file was moved from");
    return *impl_;
}

ReadOnlyFile ReadOnlyFile::open(const std::filesystem::path& path, bool follow_symlinks) {
    auto impl = std::make_unique<Impl>();
    impl->path = path;
    impl->display = display_path(path);
    impl->follow = follow_symlinks;
    const auto& display = impl->display;
    // The OS would silently open the prefix before the NUL, a different file than the one named.
    if(path.native().find(fs::path::value_type()) != fs::path::string_type::npos)
        throw std::invalid_argument(display + ": path contains a NUL character");
    std::error_code unresolved;
    impl->resolved = path.empty() || path.is_absolute() ? path : fs::absolute(path, unresolved);
    if(unresolved) impl->resolved = path;   // no current directory: resolve relative to it at each use
    const auto& resolved = impl->resolved;
    const auto refuse_link = [&] {
        throw std::system_error(std::make_error_code(std::errc::too_many_symbolic_link_levels), display + ": open: symbolic link refused");
    };
#ifdef _WIN32
    const DWORD flags = FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN | (follow_symlinks ? 0 : FILE_FLAG_OPEN_REPARSE_POINT);
    const HANDLE handle = CreateFileW(resolved.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, flags, nullptr);
    if(handle == INVALID_HANDLE_VALUE) {
        // Without FILE_FLAG_BACKUP_SEMANTICS (which can bypass ACLs) a directory fails as access denied.
        const auto error = last_error();
        if(error.value() == int(ERROR_ACCESS_DENIED)) {
            const DWORD attributes = GetFileAttributesW(resolved.c_str());
            if(attributes != INVALID_FILE_ATTRIBUTES) {
                if(!follow_symlinks && (attributes & FILE_ATTRIBUTE_REPARSE_POINT)) refuse_link();
                if(attributes & FILE_ATTRIBUTE_DIRECTORY) throw std::invalid_argument(display + ": not a regular file");
            }
        }
        fail_os(display, "open", error);
    }
    impl->handle = Handle(handle);
    FILE_ATTRIBUTE_TAG_INFO tag{};
    if(!GetFileInformationByHandleEx(handle, FileAttributeTagInfo, &tag, DWORD(sizeof tag))) fail_os(display, "stat", last_error());
    if(!follow_symlinks && (tag.FileAttributes & FILE_ATTRIBUTE_REPARSE_POINT)) refuse_link();
    if(GetFileType(handle) != FILE_TYPE_DISK || (tag.FileAttributes & FILE_ATTRIBUTE_DIRECTORY))
        throw std::invalid_argument(display + ": not a regular file");
    impl->identity = identity_of(handle, display);
#else
    // O_NONBLOCK keeps a FIFO from blocking the open; it has no effect on regular files.
    const int fd = open_retry(resolved.c_str(), O_RDONLY | O_CLOEXEC | O_NOCTTY | O_NONBLOCK | (follow_symlinks ? 0 : O_NOFOLLOW));
    if(fd < 0) {
        const auto error = last_error();
        if(!follow_symlinks && (error.value() == ELOOP || error.value() == EMLINK)) refuse_link();
        fail_os(display, "open", error);
    }
    impl->handle = Handle(fd);
    struct stat status{};
    if(::fstat(fd, &status) != 0) fail_os(display, "stat", last_error());
    if(!S_ISREG(status.st_mode)) throw std::invalid_argument(display + ": not a regular file");
    const int mode = ::fcntl(fd, F_GETFL);
    if(mode != -1) ::fcntl(fd, F_SETFL, mode & ~O_NONBLOCK);
    impl->identity = identity_of(status);
#endif
    return ReadOnlyFile(std::move(impl));
}
const std::filesystem::path& ReadOnlyFile::path() const noexcept { return impl_ ? impl_->path : empty_path; }
const FileIdentity& ReadOnlyFile::identity() const noexcept { return impl_ ? impl_->identity : empty_identity; }
FileIdentity ReadOnlyFile::handle_identity() const { return live().handle_identity(); }
FileIdentity ReadOnlyFile::path_identity() const {
    const auto& f = live();
#ifdef _WIN32
    const DWORD flags = FILE_FLAG_BACKUP_SEMANTICS | (f.follow ? 0 : FILE_FLAG_OPEN_REPARSE_POINT);
    const HANDLE handle = CreateFileW(f.resolved.c_str(), FILE_READ_ATTRIBUTES, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                      nullptr, OPEN_EXISTING, flags, nullptr);
    if(handle == INVALID_HANDLE_VALUE) fail_os(f.display, "stat", last_error());
    const Handle guard(handle);
    return identity_of(handle, f.display);
#else
    struct stat status{};
    if((f.follow ? ::stat(f.resolved.c_str(), &status) : ::lstat(f.resolved.c_str(), &status)) != 0) fail_os(f.display, "stat", last_error());
    return identity_of(status);
#endif
}
void ReadOnlyFile::verify_unchanged() const {
    const auto& f = live();
    const auto held = f.handle_identity();
    if(held != f.identity) throw std::runtime_error(f.display + ": file changed since open (" + differences(f.identity, held) + ")");
    FileIdentity named;
    try { named = path_identity(); }
    catch(const std::system_error& e) {
        throw std::runtime_error(f.display + ": file changed since open (path no longer resolves: " + e.code().message() + ")");
    }
    if(named != f.identity) throw std::runtime_error(f.display + ": file changed since open (path: " + differences(f.identity, named) + ")");
}
void ReadOnlyFile::read_exact(std::uint64_t offset, std::span<std::byte> dst) const {
    const auto& f = live();
    const auto size = f.identity.size;
    if(offset > size || dst.size() > size - offset)
        throw std::out_of_range(f.display + ": read of " + std::to_string(dst.size()) + " bytes at offset " + std::to_string(offset) +
                                " extends past end of file (size " + std::to_string(size) + ")");
    const auto got = read_at(f.handle.get(), dst.data(), dst.size(), offset, false);
    if(got.error) fail_os(f.display, "read", got.error);
    if(got.bytes < dst.size()) throw std::runtime_error(f.display + ": unexpected end of file at offset " + std::to_string(offset + got.bytes));
}
ReadStats ReadOnlyFile::read_chunks(std::uint64_t offset, std::uint64_t length, const ChunkConsumer& consume,
                                    const ReadOptions& options) const {
    const auto& f = live();
    const auto a = options.alignment;
    if(a == 0 || (a & (a - 1)) || options.chunk_bytes == 0 || options.chunk_bytes % a)
        throw std::invalid_argument("chunk_bytes must be a positive multiple of a power-of-two alignment");
    if(offset > std::numeric_limits<std::uint64_t>::max() - length) throw std::invalid_argument("region overflows");
    if(!consume) throw std::invalid_argument("chunk consumer is empty");
    if(offset + length > f.identity.size)
        throw std::out_of_range(f.display + ": region [" + std::to_string(offset) + ", " + std::to_string(offset + length) +
                                ") extends past end of file (size " + std::to_string(f.identity.size) + ")");
    ReadStats stats;
    if(length == 0) return stats;
    std::error_code refused;
    if(options.direct) {
        const Handle direct = f.open_direct(stats.fallback_reason);
        if(direct) {
            stats.direct_used = true;
            if(f.stream(direct.get(), true, offset, length, consume, options, stats, refused)) return stats;
            stats.direct_used = false;
            stats.fallback_reason = "direct read refused: " + refused.message();
        }
    }
    f.stream(f.handle.get(), false, offset, length, consume, options, stats, refused);
    return stats;
}
std::span<const std::byte> ReadOnlyFile::map() const {
    const auto view = live().mapped();
    return {view->data, view->bytes};
}
std::shared_ptr<const void> ReadOnlyFile::mapping_owner() const { return live().mapped(); }
void ReadOnlyFile::advise_willneed(std::uint64_t offset, std::uint64_t length) const noexcept {
    if(!impl_ || offset >= impl_->identity.size) return;
    length = std::min(length, impl_->identity.size - offset);
    if(length == 0) return;
    std::shared_ptr<const Mapping> view;
    try { std::lock_guard lock(impl_->mapping_mutex); view = impl_->mapping; }
    catch(...) { return; }
#ifdef _WIN32
    if(!view || !view->data) return;   // hints apply to the mapped view only
    struct MemoryRange { void* address; SIZE_T bytes; };
    using Prefetch = BOOL(WINAPI*)(HANDLE, ULONG_PTR, MemoryRange*, ULONG);
    const HMODULE kernel = GetModuleHandleW(L"kernel32.dll");
    const FARPROC address = kernel ? GetProcAddress(kernel, "PrefetchVirtualMemory") : nullptr;   // Windows 8 and later
    if(!address) return;
    Prefetch prefetch = nullptr;
    static_assert(sizeof(prefetch) == sizeof(address));
    std::memcpy(&prefetch, &address, sizeof(prefetch));
    MemoryRange range{const_cast<std::byte*>(view->data) + std::size_t(offset), SIZE_T(length)};
    prefetch(GetCurrentProcess(), 1, &range, 0);
#else
    if(view && view->data) {
        const long page_size = ::sysconf(_SC_PAGESIZE);
        const std::uint64_t page = page_size > 0 ? std::uint64_t(page_size) : 4096;
        const std::uint64_t begin = offset / page * page;
        ::madvise(const_cast<std::byte*>(view->data) + std::size_t(begin), std::size_t(offset - begin + length), MADV_WILLNEED);
        return;
    }
#if defined(POSIX_FADV_WILLNEED)
    if(offset <= std::uint64_t(std::numeric_limits<off_t>::max()) && length <= std::uint64_t(std::numeric_limits<off_t>::max()))
        ::posix_fadvise(impl_->handle.get(), off_t(offset), off_t(length), POSIX_FADV_WILLNEED);
#elif defined(F_RDADVISE)
    if(offset <= std::uint64_t(std::numeric_limits<off_t>::max())) {
        struct radvisory advice{};
        advice.ra_offset = off_t(offset);
        advice.ra_count = int(std::min<std::uint64_t>(length, std::uint64_t(std::numeric_limits<int>::max())));
        ::fcntl(impl_->handle.get(), F_RDADVISE, &advice);
    }
#endif
#endif
}
}
