// lookup-table-rows' native row reader for its device form (device_rows.py), built at load by torch's extension
// builder (row_reader.py); host code only. A table of fixed-size rows over one or more files (each holding the next
// rows in id order) opened for random access (POSIX_FADV_RANDOM: no read-ahead), the rows a gather needs read with
// positioned reads by a persistent pool of threads that take rows from an atomic counter (no lock per row), rows asked
// for twice read once, an optional least-recently-used cache of hot rows that prefetch() fills in the background, and
// calls serialized (one gather, or one slice of a prefetch batch, at a time: a gather never waits behind a whole cold
// prefetch). A gather that fails inside a host function zero-fills its rows and marks the reader failed; the mark
// stays, and the Python side refuses every later lookup. enqueue_gather puts a gather on a CUDA stream as a host
// function (cudaLaunchHostFunc, which a CUDA graph captures as a host node): the ids a device step wrote into pinned
// memory are read there and the rows written into pinned memory before the stream goes on, so a captured round reads
// file rows without Python.
#include <torch/extension.h>
#if defined(CRIT_ROWS_CUDA)
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#endif

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <list>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

struct File {
  int fd = -1;
  uint64_t offset = 0, rows = 0, first = 0;   // its rows are ids [first, first + rows), at byte offset in the file
};

struct Job {
  uint64_t id;
  uint8_t* dst;
};

class Reader {
 public:
  Reader(const std::vector<std::string>& paths, const std::vector<int64_t>& offsets, const std::vector<int64_t>& rows,
         int64_t row_bytes, int64_t threads, int64_t cache_rows)
      : row_bytes_(static_cast<size_t>(row_bytes)), cache_rows_(static_cast<size_t>(cache_rows)) {
    if (paths.empty() || paths.size() != offsets.size() || paths.size() != rows.size() || row_bytes < 1 || threads < 1)
      throw std::invalid_argument("row reader: files, their offsets and row counts, a row size and threads");
    uint64_t first = 0;
    for (size_t i = 0; i < paths.size(); ++i) {
      File f;
      f.fd = ::open(paths[i].c_str(), O_RDONLY | O_CLOEXEC);
      if (f.fd < 0) throw std::runtime_error("row reader: cannot open " + paths[i]);
      struct stat st;
      if (::fstat(f.fd, &st) != 0) throw std::runtime_error("row reader: cannot stat " + paths[i]);
      f.offset = static_cast<uint64_t>(offsets[i]);
      f.rows = static_cast<uint64_t>(rows[i]);
      f.first = first;
      if (f.offset + f.rows * row_bytes_ > static_cast<uint64_t>(st.st_size))
        throw std::invalid_argument("row reader: " + paths[i] + " is shorter than its rows");
#ifdef POSIX_FADV_RANDOM
      if (::posix_fadvise(f.fd, static_cast<off_t>(f.offset), static_cast<off_t>(f.rows * row_bytes_),
                          POSIX_FADV_RANDOM) != 0)
        advised_ = false;   // the advice holds only when every file took it
#else
      advised_ = false;
#endif
      first += f.rows;
      files_.push_back(f);
    }
    rows_ = first;
    if (cache_rows_) {
      cache_.resize(cache_rows_ * row_bytes_);
      for (size_t s = 0; s < cache_rows_; ++s) free_.push_back(s);
    }
    for (int64_t t = 1; t < threads; ++t) pool_.emplace_back([this] { serve(); });
    prefetcher_ = std::thread([this] { prefetch_loop(); });
  }

  ~Reader() {
    {
      std::lock_guard<std::mutex> lock(work_mutex_);
      stop_ = true;
    }
    wake_.notify_all();
    for (auto& t : pool_) t.join();
    {
      std::lock_guard<std::mutex> lock(prefetch_mutex_);
      stop_prefetch_ = true;
    }
    prefetch_wake_.notify_all();
    prefetcher_.join();
    for (auto& f : files_)
      if (f.fd >= 0) ::close(f.fd);
  }

  uint64_t rows() const { return rows_; }
  size_t row_bytes() const { return row_bytes_; }
  std::string advice() const { return advised_ ? "fadvise_random" : "none"; }

  // Rows ids[i] at out + i * row_bytes, in order; an id outside the table throws before anything is read.
  void gather(const int64_t* ids, int64_t n, uint8_t* out) {
    for (int64_t i = 0; i < n; ++i)
      if (ids[i] < 0 || static_cast<uint64_t>(ids[i]) >= rows_) throw std::out_of_range("row reader: row outside the table");
    std::lock_guard<std::mutex> call(call_mutex_);
    std::unordered_map<int64_t, int64_t> first;
    first.reserve(static_cast<size_t>(n) * 2);
    std::vector<Job> misses;
    for (int64_t i = 0; i < n; ++i) {
      if (!first.emplace(ids[i], i).second) continue;
      uint8_t* dst = out + static_cast<size_t>(i) * row_bytes_;
      if (cache_rows_) {
        auto hit = where_.find(ids[i]);
        if (hit != where_.end()) {
          std::memcpy(dst, cache_.data() + hit->second.slot * row_bytes_, row_bytes_);
          lru_.splice(lru_.end(), lru_, hit->second.it);
          hits_.fetch_add(1);
          continue;
        }
      }
      misses.push_back({static_cast<uint64_t>(ids[i]), dst});
    }
    run(misses);
    for (const auto& job : misses) remember(static_cast<int64_t>(job.id), job.dst);
    for (int64_t i = 0; i < n; ++i) {
      const int64_t j = first[ids[i]];
      if (j != i) std::memcpy(out + static_cast<size_t>(i) * row_bytes_, out + static_cast<size_t>(j) * row_bytes_, row_bytes_);
    }
  }

  // Starts reading ids into the hot-row cache in the background (rows already held are skipped); a later gather waits
  // for a batch in flight, then finds its rows held.
  void prefetch(const int64_t* ids, int64_t n) {
    if (!cache_rows_) return;
    std::vector<int64_t> wanted;
    for (int64_t i = 0; i < n; ++i)
      if (ids[i] >= 0 && static_cast<uint64_t>(ids[i]) < rows_) wanted.push_back(ids[i]);
    {
      std::lock_guard<std::mutex> lock(prefetch_mutex_);
      pending_.insert(pending_.end(), wanted.begin(), wanted.end());
    }
    prefetch_wake_.notify_one();
  }

  // Blocks until every prefetch queued so far has landed.
  void drain() {
    std::unique_lock<std::mutex> lock(prefetch_mutex_);
    prefetch_done_.wait(lock, [this] { return pending_.empty() && !prefetching_; });
  }

  std::vector<int64_t> stats() const {
    return {static_cast<int64_t>(reads_.load()), static_cast<int64_t>(hits_.load()),
            static_cast<int64_t>(prefetched_.load()), static_cast<int64_t>(failures_.load())};
  }

  void count_failure() {
    ++failures_;
    failed_.store(true);
  }
  bool failed() const { return failed_.load(); }

 private:
  struct Held {
    size_t slot;
    std::list<int64_t>::iterator it;
  };

  void read_one(const Job& job) {
    const File* f = &files_.back();
    for (const auto& file : files_)
      if (job.id < file.first + file.rows) {
        f = &file;
        break;
      }
    const uint64_t at = f->offset + (job.id - f->first) * row_bytes_;
    size_t done = 0;
    while (done < row_bytes_) {
      const ssize_t got = ::pread(f->fd, job.dst + done, row_bytes_ - done, static_cast<off_t>(at + done));
      if (got < 0) {
        if (errno == EINTR) continue;
        throw std::runtime_error("row reader: read failed");
      }
      if (got == 0) throw std::runtime_error("row reader: short read");
      done += static_cast<size_t>(got);
    }
    ++reads_;
  }

  // Reads every job with the pool and the calling thread; rethrows the first failure.
  void run(const std::vector<Job>& jobs) {
    if (jobs.empty()) return;
    if (pool_.empty() || jobs.size() == 1) {
      for (const auto& job : jobs) read_one(job);
      return;
    }
    {
      std::lock_guard<std::mutex> lock(work_mutex_);
      jobs_ = &jobs;
      next_.store(0);
      left_ = jobs.size();
      failure_ = nullptr;
      ++generation_;
    }
    wake_.notify_all();
    work();
    std::unique_lock<std::mutex> lock(work_mutex_);
    done_.wait(lock, [this] { return left_ == 0 && active_ == 0; });
    jobs_ = nullptr;
    if (failure_) std::rethrow_exception(failure_);
  }

  // A thread joins the batch in flight under the lock (active_), so the caller never ends a batch a thread still reads.
  void work() {
    const std::vector<Job>* jobs;
    {
      std::lock_guard<std::mutex> lock(work_mutex_);
      jobs = jobs_;
      if (!jobs) return;
      ++active_;
    }
    size_t finished = 0;
    std::exception_ptr error;
    for (;;) {
      const size_t i = next_.fetch_add(1);
      if (i >= jobs->size()) break;
      try {
        read_one((*jobs)[i]);
      } catch (...) {
        if (!error) error = std::current_exception();
      }
      ++finished;
    }
    std::lock_guard<std::mutex> lock(work_mutex_);
    if (error && !failure_) failure_ = error;
    left_ -= finished;
    --active_;
    if (left_ == 0 && active_ == 0) done_.notify_all();
  }

  void serve() {
    uint64_t seen = 0;
    for (;;) {
      {
        std::unique_lock<std::mutex> lock(work_mutex_);
        wake_.wait(lock, [&] { return stop_ || (generation_ != seen && jobs_); });
        if (stop_) return;
        seen = generation_;
      }
      work();
    }
  }

  void remember(int64_t id, const uint8_t* row) {
    if (!cache_rows_ || where_.count(id)) return;
    size_t slot;
    if (!free_.empty()) {
      slot = free_.front();
      free_.pop_front();
    } else {
      const int64_t old = lru_.front();
      lru_.pop_front();
      slot = where_[old].slot;
      where_.erase(old);
    }
    std::memcpy(cache_.data() + slot * row_bytes_, row, row_bytes_);
    lru_.push_back(id);
    where_[id] = Held{slot, std::prev(lru_.end())};
  }

  void prefetch_loop() {
    constexpr size_t kSlice = 256;   // rows read under one hold of the call lock
    std::vector<uint8_t> buffer;
    for (;;) {
      std::vector<int64_t> batch;
      {
        std::unique_lock<std::mutex> lock(prefetch_mutex_);
        prefetch_wake_.wait(lock, [this] { return stop_prefetch_ || !pending_.empty(); });
        if (stop_prefetch_ && pending_.empty()) return;
        batch.swap(pending_);
        prefetching_ = true;
      }
      std::unordered_map<int64_t, bool> once;
      size_t taken = 0;
      for (size_t at = 0; at < batch.size() && taken < cache_rows_; at += kSlice) {
        std::lock_guard<std::mutex> call(call_mutex_);
        std::vector<int64_t> wanted;
        for (size_t i = at; i < batch.size() && i < at + kSlice; ++i)
          if (!where_.count(batch[i]) && once.emplace(batch[i], true).second) wanted.push_back(batch[i]);
        if (wanted.size() > cache_rows_ - taken) wanted.resize(cache_rows_ - taken);
        taken += wanted.size();
        buffer.resize(wanted.size() * row_bytes_);
        std::vector<Job> jobs;
        for (size_t i = 0; i < wanted.size(); ++i) jobs.push_back({static_cast<uint64_t>(wanted[i]), buffer.data() + i * row_bytes_});
        try {
          run(jobs);
          for (size_t i = 0; i < wanted.size(); ++i) remember(wanted[i], buffer.data() + i * row_bytes_);
          prefetched_.fetch_add(wanted.size());
        } catch (...) {
          ++failures_;
        }
      }
      {
        std::lock_guard<std::mutex> lock(prefetch_mutex_);
        prefetching_ = false;
      }
      prefetch_done_.notify_all();
    }
  }

  std::vector<File> files_;
  uint64_t rows_ = 0;
  size_t row_bytes_, cache_rows_;
  bool advised_ = true;
  // the pool
  std::vector<std::thread> pool_;
  std::mutex work_mutex_;
  std::condition_variable wake_, done_;
  const std::vector<Job>* jobs_ = nullptr;
  std::atomic<size_t> next_{0};
  size_t left_ = 0, active_ = 0;
  uint64_t generation_ = 0;
  std::exception_ptr failure_;
  bool stop_ = false;
  // one gather or prefetch batch at a time
  std::mutex call_mutex_;
  // the hot-row cache (held under call_mutex_)
  std::vector<uint8_t> cache_;
  std::deque<size_t> free_;
  std::list<int64_t> lru_;
  std::unordered_map<int64_t, Held> where_;
  std::atomic<uint64_t> hits_{0}, prefetched_{0}, reads_{0}, failures_{0};
  std::atomic<bool> failed_{false};
  // prefetch
  std::thread prefetcher_;
  std::mutex prefetch_mutex_;
  std::condition_variable prefetch_wake_, prefetch_done_;
  std::vector<int64_t> pending_;
  bool prefetching_ = false, stop_prefetch_ = false;
};

// A gather a CUDA stream runs as a host function: ids and rows in pinned host memory, which the site holds (a captured
// graph's host node may run after the Python side let go of them).
struct Site {
  std::shared_ptr<Reader> reader;
  torch::Tensor ids_tensor, out_tensor;
  const int64_t* ids;
  int64_t n;
  uint8_t* out;
};

#if defined(CRIT_ROWS_CUDA)
void CUDART_CB run_site(void* data) {
  auto* site = static_cast<Site*>(data);
  try {
    site->reader->gather(site->ids, site->n, site->out);
  } catch (...) {
    std::memset(site->out, 0, static_cast<size_t>(site->n) * site->reader->row_bytes());
    site->reader->count_failure();
  }
}
#endif

class Handle {
 public:
  Handle(std::vector<std::string> paths, std::vector<int64_t> offsets, std::vector<int64_t> rows, int64_t row_bytes,
         int64_t threads, int64_t cache_rows)
      : reader_(std::make_shared<Reader>(paths, offsets, rows, row_bytes, threads, cache_rows)) {}

  void gather(torch::Tensor ids, torch::Tensor out) {
    TORCH_CHECK(ids.device().is_cpu() && ids.scalar_type() == torch::kInt64 && ids.is_contiguous(), "ids: int64 CPU");
    TORCH_CHECK(out.device().is_cpu() && out.scalar_type() == torch::kUInt8 && out.is_contiguous() &&
                out.numel() >= ids.numel() * static_cast<int64_t>(reader_->row_bytes()), "out: uint8 CPU rows");
    const int64_t* p = ids.data_ptr<int64_t>();
    uint8_t* o = out.data_ptr<uint8_t>();
    const int64_t n = ids.numel();
    pybind11::gil_scoped_release release;
    reader_->gather(p, n, o);
  }

  void prefetch(torch::Tensor ids) {
    TORCH_CHECK(ids.device().is_cpu() && ids.scalar_type() == torch::kInt64 && ids.is_contiguous(), "ids: int64 CPU");
    reader_->prefetch(ids.data_ptr<int64_t>(), ids.numel());
  }

  void drain() {
    pybind11::gil_scoped_release release;
    reader_->drain();
  }

  // A site for enqueue(): ids [n] int64 and out [n][row_bytes] uint8, pinned, kept alive by the caller.
  int64_t site(torch::Tensor ids, torch::Tensor out) {
    TORCH_CHECK(ids.is_pinned() && out.is_pinned(), "a site's ids and rows are pinned host memory");
    sites_.push_back(std::make_unique<Site>(Site{reader_, ids, out, ids.data_ptr<int64_t>(), ids.numel(),
                                                 out.data_ptr<uint8_t>()}));
    return static_cast<int64_t>(sites_.size() - 1);
  }

  void enqueue(int64_t site, int64_t stream) {
    TORCH_CHECK(site >= 0 && site < static_cast<int64_t>(sites_.size()), "no such site");
#if defined(CRIT_ROWS_CUDA)
    C10_CUDA_CHECK(cudaLaunchHostFunc(reinterpret_cast<cudaStream_t>(stream), run_site, sites_[site].get()));
#else
    TORCH_CHECK(false, "this row reader was built without CUDA: no host functions on a stream");
#endif
  }

  int64_t rows() const { return static_cast<int64_t>(reader_->rows()); }
  std::string advice() const { return reader_->advice(); }
  std::vector<int64_t> stats() const { return reader_->stats(); }
  bool failed() const { return reader_->failed(); }

 private:
  std::shared_ptr<Reader> reader_;
  std::vector<std::unique_ptr<Site>> sites_;
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  pybind11::class_<Handle>(m, "Reader")
      .def(pybind11::init<std::vector<std::string>, std::vector<int64_t>, std::vector<int64_t>, int64_t, int64_t,
                          int64_t>())
      .def("gather", &Handle::gather)
      .def("prefetch", &Handle::prefetch)
      .def("drain", &Handle::drain)
      .def("site", &Handle::site)
      .def("enqueue", &Handle::enqueue)
      .def("rows", &Handle::rows)
      .def("advice", &Handle::advice)
      .def("stats", &Handle::stats)
      .def("failed", &Handle::failed);
}
