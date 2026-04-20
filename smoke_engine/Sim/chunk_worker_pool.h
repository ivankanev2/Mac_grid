#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

class ChunkWorkerPool {
public:
    ChunkWorkerPool() {
        const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
        unsigned cappedHw = hw;
#if defined(__APPLE__)
        cappedHw = std::min(cappedHw, 8u);
#endif
        if (const char* envWorkers = std::getenv("SMOKE_MAX_WORKERS")) {
            char* endPtr = nullptr;
            const unsigned long parsed = std::strtoul(envWorkers, &endPtr, 10);
            if (endPtr != envWorkers && parsed > 0ul) {
                cappedHw = std::max(1u, std::min(hw, (unsigned)parsed));
            }
        }
        const unsigned backgroundCount = (cappedHw > 1u) ? (cappedHw - 1u) : 0u;
        m_workers.reserve(backgroundCount);
        for (unsigned workerIndex = 0; workerIndex < backgroundCount; ++workerIndex) {
            m_workers.emplace_back([this, workerIndex]() { workerLoop((int)workerIndex); });
        }
    }

    ~ChunkWorkerPool() {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_stop = true;
            ++m_epoch;
        }
        m_workCv.notify_all();
        for (std::thread& worker : m_workers) {
            if (worker.joinable()) worker.join();
        }
    }

    ChunkWorkerPool(const ChunkWorkerPool&) = delete;
    ChunkWorkerPool& operator=(const ChunkWorkerPool&) = delete;

    int maxWorkers() const {
        return 1 + (int)m_workers.size();
    }

    template <typename Fn>
    void parallelFor(int count, int minChunk, Fn&& fn) {
        if (count <= 0) return;

        const int safeMinChunk = std::max(1, minChunk);
        const int totalWorkers = maxWorkers();
        const int maxUsefulWorkers = std::min(totalWorkers, std::max(1, (count + safeMinChunk - 1) / safeMinChunk));
        if (maxUsefulWorkers <= 1 || m_workers.empty()) {
            fn(0, count);
            return;
        }

        const int desiredChunks = std::max(maxUsefulWorkers, maxUsefulWorkers * 2);
        const int chunkSize = std::max(safeMinChunk, (count + desiredChunks - 1) / desiredChunks);
        const int chunkCount = (count + chunkSize - 1) / chunkSize;
        if (chunkCount <= 1) {
            fn(0, count);
            return;
        }

        runParallel(count, chunkSize, std::function<void(int, int)>(std::forward<Fn>(fn)));
    }

private:
    void runParallel(int count, int chunkSize, std::function<void(int, int)> fn) {
        const int activeWorkers = std::min((int)m_workers.size(), std::max(0, (count + chunkSize - 1) / chunkSize - 1));
        if (activeWorkers <= 0) {
            fn(0, count);
            return;
        }

        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_jobCount = count;
            m_chunkSize = chunkSize;
            m_job = std::move(fn);
            m_activeWorkers = activeWorkers;
            m_finishedWorkers = 0;
            m_nextChunk.store(0, std::memory_order_release);
            ++m_epoch;
        }

        m_workCv.notify_all();
        executeChunks();

        std::unique_lock<std::mutex> lock(m_mutex);
        m_doneCv.wait(lock, [this]() { return m_finishedWorkers >= m_activeWorkers; });
        m_job = nullptr;
        m_activeWorkers = 0;
        m_finishedWorkers = 0;
    }

    void workerLoop(int workerIndex) {
        unsigned localEpoch = 0;
        for (;;) {
            std::function<void(int, int)> localJob;
            int localJobCount = 0;
            int localChunkSize = 1;
            bool participate = false;

            {
                std::unique_lock<std::mutex> lock(m_mutex);
                m_workCv.wait(lock, [this, localEpoch]() { return m_stop || m_epoch != localEpoch; });
                if (m_stop) return;

                localEpoch = m_epoch;
                participate = workerIndex < m_activeWorkers;
                if (participate) {
                    localJob = m_job;
                    localJobCount = m_jobCount;
                    localChunkSize = m_chunkSize;
                }
            }

            if (participate && localJob) {
                for (;;) {
                    const int chunkIndex = m_nextChunk.fetch_add(1, std::memory_order_relaxed);
                    const int begin = chunkIndex * localChunkSize;
                    if (begin >= localJobCount) break;
                    localJob(begin, std::min(localJobCount, begin + localChunkSize));
                }
            }

            if (participate) {
                std::lock_guard<std::mutex> lock(m_mutex);
                ++m_finishedWorkers;
                if (m_finishedWorkers >= m_activeWorkers) {
                    m_doneCv.notify_one();
                }
            }
        }
    }

    void executeChunks() {
        std::function<void(int, int)> localJob;
        int localJobCount = 0;
        int localChunkSize = 1;

        {
            std::lock_guard<std::mutex> lock(m_mutex);
            localJob = m_job;
            localJobCount = m_jobCount;
            localChunkSize = m_chunkSize;
        }

        if (!localJob) return;
        for (;;) {
            const int chunkIndex = m_nextChunk.fetch_add(1, std::memory_order_relaxed);
            const int begin = chunkIndex * localChunkSize;
            if (begin >= localJobCount) break;
            localJob(begin, std::min(localJobCount, begin + localChunkSize));
        }
    }

    std::vector<std::thread> m_workers;
    std::mutex m_mutex;
    std::condition_variable m_workCv;
    std::condition_variable m_doneCv;
    std::function<void(int, int)> m_job;
    std::atomic<int> m_nextChunk{0};
    int m_jobCount = 0;
    int m_chunkSize = 1;
    int m_activeWorkers = 0;
    int m_finishedWorkers = 0;
    unsigned m_epoch = 0;
    bool m_stop = false;
};

inline ChunkWorkerPool& sharedChunkWorkerPool() {
    static ChunkWorkerPool pool;
    return pool;
}
