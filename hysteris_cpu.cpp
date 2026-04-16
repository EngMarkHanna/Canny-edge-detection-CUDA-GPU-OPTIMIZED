// Scalar oracle + two OpenMP hysteresis variants (basic, NUMA-aware).
//
// Correctness invariant (shared by both OMP variants):
//   The only write to `map` is 0 -> 2. Single-byte stores are atomic on x86
//   w.r.t. other threads' single-byte reads, so the CAS is only used to
//   deduplicate frontier enqueues — it is NOT load-bearing for the map's
//   final value. Two threads racing to promote the same pixel both write 2,
//   and exactly one of them wins the CAS (the other does not push a dup).
//
// Bit-exact vs scalar:
//   Hysteresis' output is uniquely defined by the input (every 0 reachable
//   from a 2 through 0s becomes 2). Order of traversal doesn't change it.

#include "hysteris_cpu.hpp"

#include <vector>
#include <cstdint>
#include <omp.h>

// ─────────────────────────────────────────────────────────────────────────────
// Portable single-byte CAS. Returns true if *p was `expected` and is now `desired`.
// ─────────────────────────────────────────────────────────────────────────────
#if defined(_MSC_VER)
  #include <intrin.h>
    static inline bool cas_u8(unsigned char* p, unsigned char expected, unsigned char desired) {
          char prev = _InterlockedCompareExchange8((char*)p, (char)desired, (char)expected);
                return (unsigned char)prev == expected;
                  }
                  #else
                    static inline bool cas_u8(unsigned char* p, unsigned char expected, unsigned char desired) {
                          return __atomic_compare_exchange_n(p, &expected, desired,
                                                                   false,
                                                                                                            __ATOMIC_RELAXED, __ATOMIC_RELAXED);
                                                                                                              }
                                                                                                              #endif

// ─────────────────────────────────────────────────────────────────────────────
// Scalar oracle. DFS via back-of-vector pop to avoid std::deque overhead.
// ─────────────────────────────────────────────────────────────────────────────
void hysteresis_scalar(unsigned char* map, int mapW, int mapH)
{
    std::vector<unsigned char*> stack;
    stack.reserve(1 << 16);
    for (int y = 1; y < mapH - 1; ++y) {
        unsigned char* row = map + y * mapW;
        for (int x = 1; x < mapW - 1; ++x)
            if (row[x] == 2)
                stack.push_back(row + x);
    }

    const int s = mapW;
    while (!stack.empty()) {
        unsigned char* m = stack.back();
        stack.pop_back();
        if (!m[-s - 1]) { m[-s - 1] = 2; stack.push_back(m - s - 1); }
        if (!m[-s    ]) { m[-s    ] = 2; stack.push_back(m - s    ); }
        if (!m[-s + 1]) { m[-s + 1] = 2; stack.push_back(m - s + 1); }
        if (!m[-1    ]) { m[-1    ] = 2; stack.push_back(m - 1    ); }
        if (!m[ 1    ]) { m[ 1    ] = 2; stack.push_back(m + 1    ); }
        if (!m[ s - 1]) { m[ s - 1] = 2; stack.push_back(m + s - 1); }
        if (!m[ s    ]) { m[ s    ] = 2; stack.push_back(m + s    ); }
        if (!m[ s + 1]) { m[ s + 1] = 2; stack.push_back(m + s + 1); }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Final output: map interior (W+2)x(H+2) -> binary WxH
// ─────────────────────────────────────────────────────────────────────────────
void final_output(const unsigned char* map, int mapW,
                  unsigned char* dst, int W, int H)
{
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < H; ++y) {
        const unsigned char* pmap = map + (y + 1) * mapW + 1;
        unsigned char* pdst = dst + y * W;
        for (int x = 0; x < W; ++x)
            pdst[x] = (pmap[x] == 2) ? 255 : 0;
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// BASIC: per-thread local queue, CAS-dedup enqueue, simple drain.
// ═════════════════════════════════════════════════════════════════════════════
void hysteresis_omp_basic(unsigned char* map, int mapW, int mapH, int num_threads)
{
    if (num_threads > 0) omp_set_num_threads(num_threads);
    const int s = mapW;

    #pragma omp parallel
    {
        std::vector<unsigned char*> q;
        q.reserve(1 << 14);

        #pragma omp for schedule(static) nowait
        for (int y = 1; y < mapH - 1; ++y) {
            unsigned char* row = map + y * mapW;
            for (int x = 1; x < mapW - 1; ++x)
                if (row[x] == 2) q.push_back(row + x);
        }
        #pragma omp barrier

        while (!q.empty()) {
            unsigned char* m = q.back(); q.pop_back();
            if (m[-s - 1] == 0 && cas_u8(m - s - 1, 0, 2)) q.push_back(m - s - 1);
            if (m[-s    ] == 0 && cas_u8(m - s,     0, 2)) q.push_back(m - s    );
            if (m[-s + 1] == 0 && cas_u8(m - s + 1, 0, 2)) q.push_back(m - s + 1);
            if (m[-1    ] == 0 && cas_u8(m - 1,     0, 2)) q.push_back(m - 1    );
            if (m[ 1    ] == 0 && cas_u8(m + 1,     0, 2)) q.push_back(m + 1    );
            if (m[ s - 1] == 0 && cas_u8(m + s - 1, 0, 2)) q.push_back(m + s - 1);
            if (m[ s    ] == 0 && cas_u8(m + s,     0, 2)) q.push_back(m + s    );
            if (m[ s + 1] == 0 && cas_u8(m + s + 1, 0, 2)) q.push_back(m + s + 1);
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// NUMA: same BFS body as basic, with two NUMA-aware tweaks:
//
//   1. proc_bind(close) — on a 2-socket Xeon Silver 4208 with default
//      OMP_PLACES=cores, `close` packs threads onto the first socket before
//      spilling to the second. At T=8 the whole team runs on one socket,
//      which keeps the shared `map` hot in that socket's 11MB L3 and avoids
//      UPI/QPI coherence traffic on every CAS.
//
//   2. Per-thread queues are reserved inside the parallel region, so OS
//      first-touch places those pages on the executing thread's NUMA node.
//
// Algorithm and bit-exact output are identical to the basic variant — only
// the placement changes.
// ═════════════════════════════════════════════════════════════════════════════
void hysteresis_omp_numa(unsigned char* map, int mapW, int mapH, int num_threads)
{
    if (num_threads > 0) omp_set_num_threads(num_threads);
    const int s = mapW;

    #pragma omp parallel proc_bind(close)
    {
        std::vector<unsigned char*> q;
        q.reserve(1 << 14);

        #pragma omp for schedule(static) nowait
        for (int y = 1; y < mapH - 1; ++y) {
            unsigned char* row = map + y * mapW;
            for (int x = 1; x < mapW - 1; ++x)
                if (row[x] == 2) q.push_back(row + x);
        }
        #pragma omp barrier

        while (!q.empty()) {
            unsigned char* m = q.back(); q.pop_back();
            if (m[-s - 1] == 0 && cas_u8(m - s - 1, 0, 2)) q.push_back(m - s - 1);
            if (m[-s    ] == 0 && cas_u8(m - s,     0, 2)) q.push_back(m - s    );
            if (m[-s + 1] == 0 && cas_u8(m - s + 1, 0, 2)) q.push_back(m - s + 1);
            if (m[-1    ] == 0 && cas_u8(m - 1,     0, 2)) q.push_back(m - 1    );
            if (m[ 1    ] == 0 && cas_u8(m + 1,     0, 2)) q.push_back(m + 1    );
            if (m[ s - 1] == 0 && cas_u8(m + s - 1, 0, 2)) q.push_back(m + s - 1);
            if (m[ s    ] == 0 && cas_u8(m + s,     0, 2)) q.push_back(m + s    );
            if (m[ s + 1] == 0 && cas_u8(m + s + 1, 0, 2)) q.push_back(m + s + 1);
        }
    }
}
