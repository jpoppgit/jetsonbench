// CPU-only vs NEON vs NEON+OpenMP on Jetson (aarch64)
// Kernels: dot product (memory-bound), GEMM C=A*B (compute-bound)
// Usage: ./bench [N_dot=16777216] [n_gemm=768]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

// True "scalar": disable auto-vectorization for these functions
#define SCALAR_FN __attribute__((noinline, optimize("no-tree-vectorize")))
#define FN __attribute__((noinline))

// =====================================================================
// Dot product
// =====================================================================
SCALAR_FN float dot_scalar(const float* a, const float* b, size_t n) {
    float s = 0.f;
    for (size_t i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

// Reference in double (not part of the timing)
SCALAR_FN double dot_ref(const float* a, const float* b, size_t n) {
    double s = 0.0;
    for (size_t i = 0; i < n; i++) s += (double)a[i] * (double)b[i];
    return s;
}

// GCC vectorizes float reductions only when explicitly allowed -> omp simd
FN float dot_auto(const float* a, const float* b, size_t n) {
    float s = 0.f;
#pragma omp simd reduction(+ : s)
    for (size_t i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

#ifdef __ARM_NEON
FN float dot_neon(const float* a, const float* b, size_t n) {
    float32x4_t s0 = vdupq_n_f32(0), s1 = s0, s2 = s0, s3 = s0;  // 4 accumulators: hide latency
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i));
        s1 = vfmaq_f32(s1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
        s2 = vfmaq_f32(s2, vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
        s3 = vfmaq_f32(s3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    }
    float s = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}

FN float dot_neon_omp(const float* a, const float* b, size_t n) {
    const size_t CH = 1 << 16;
    size_t chunks = (n + CH - 1) / CH;
    float total = 0.f;
#pragma omp parallel for reduction(+ : total) schedule(static)
    for (size_t c = 0; c < chunks; c++) {
        size_t o = c * CH, len = std::min(CH, n - o);
        total += dot_neon(a + o, b + o, len);
    }
    return total;
}
#endif

// =====================================================================
// GEMM: C = A * B, all n x n, row-major, float
// 2*n^3 FLOPs on 3*n^2*4 bytes -> high arithmetic intensity
// =====================================================================

// Baseline: ikj loop order (row-wise access), but no SIMD
SCALAR_FN void gemm_scalar(const float* __restrict A, const float* __restrict B,
                           float* __restrict C, size_t n) {
    std::fill(C, C + n * n, 0.f);
    for (size_t i = 0; i < n; i++)
        for (size_t k = 0; k < n; k++) {
            float a = A[i * n + k];
            for (size_t j = 0; j < n; j++) C[i * n + j] += a * B[k * n + j];
        }
}

// Same loop; the compiler may vectorize, without blocking
FN void gemm_auto(const float* __restrict A, const float* __restrict B,
                  float* __restrict C, size_t n) {
    std::fill(C, C + n * n, 0.f);
    for (size_t i = 0; i < n; i++)
        for (size_t k = 0; k < n; k++) {
            float a = A[i * n + k];
            float* c = C + i * n;
            const float* b = B + k * n;
#pragma omp simd
            for (size_t j = 0; j < n; j++) c[j] += a * b[j];
        }
}

#ifdef __ARM_NEON
// Microkernel: 8x8 tile of C in 16 NEON registers, k loop from k0 to k1.
// Per k: 2 vector loads (B), 8 broadcasts (A), 16 FMAs.
static inline __attribute__((always_inline)) void ukernel_8x8(const float* A, const float* B,
                                                              float* C, size_t n, size_t k0,
                                                              size_t k1) {
    float32x4_t c[8][2];
#pragma GCC unroll 8
    for (int r = 0; r < 8; r++) {
        c[r][0] = vld1q_f32(C + r * n);
        c[r][1] = vld1q_f32(C + r * n + 4);
    }
    for (size_t k = k0; k < k1; k++) {
        float32x4_t b0 = vld1q_f32(B + k * n);
        float32x4_t b1 = vld1q_f32(B + k * n + 4);
#pragma GCC unroll 8
        for (int r = 0; r < 8; r++) {
            float32x4_t a = vdupq_n_f32(A[r * n + k]);
            c[r][0] = vfmaq_f32(c[r][0], b0, a);
            c[r][1] = vfmaq_f32(c[r][1], b1, a);
        }
    }
#pragma GCC unroll 8
    for (int r = 0; r < 8; r++) {
        vst1q_f32(C + r * n, c[r][0]);
        vst1q_f32(C + r * n + 4, c[r][1]);
    }
}

static const size_t KC = 256, NC = 128;  // Cache blocking (B panel KC x NC = 128 KB)

// Computes the block C[i0:i1, j0:j1]. (i1-i0) and (j1-j0) must be multiples of 8.
// Loop order: k-block outermost, so the B sub-panel (KC x (j1-j0)) is reused by
// every row block of this task -> stays in L2 instead of being re-fetched.
FN void gemm_neon_block(const float* A, const float* B, float* C, size_t n, size_t i0, size_t i1,
                        size_t j0, size_t j1) {
    for (size_t i = i0; i < i1; i++) std::fill(C + i * n + j0, C + i * n + j1, 0.f);
    for (size_t k0 = 0; k0 < n; k0 += KC) {
        size_t k1 = std::min(k0 + KC, n);
        for (size_t i = i0; i < i1; i += 8)
            for (size_t j = j0; j < j1; j += 8)
                ukernel_8x8(A + i * n, B + j, C + i * n + j, n, k0, k1);
    }
}

// Single thread: one task per column panel, covering all rows
FN void gemm_neon(const float* A, const float* B, float* C, size_t n) {
    for (size_t j0 = 0; j0 < n; j0 += NC)
        gemm_neon_block(A, B, C, n, 0, n, j0, std::min(j0 + NC, n));
}

// Multi-thread: 2D tasks (column panel x row chunk). The row chunk is chosen so that there
// are >= 4 tasks per thread (load balance) but chunks stay as tall as possible (B reuse).
// With 1 thread this degenerates to exactly gemm_neon.
FN void gemm_neon_omp(const float* A, const float* B, float* C, size_t n) {
    int nt = 1;
#ifdef _OPENMP
    nt = omp_get_max_threads();
#endif
    size_t panels = (n + NC - 1) / NC;
    size_t want = std::max<size_t>(1, (4 * (size_t)nt + panels - 1) / panels);  // row chunks
    size_t RB = (((n + want - 1) / want) + 7) / 8 * 8;                          // multiple of 8
    size_t chunks = (n + RB - 1) / RB;
#pragma omp parallel for collapse(2) schedule(static)
    for (size_t p = 0; p < panels; p++)
        for (size_t c = 0; c < chunks; c++) {
            size_t i0 = c * RB, i1 = std::min(i0 + RB, n);
            size_t j0 = p * NC, j1 = std::min(j0 + NC, n);
            gemm_neon_block(A, B, C, n, i0, i1, j0, j1);
        }
}
#endif

// =====================================================================
// Harness
// =====================================================================
// Median of `reps` samples. Fast kernels (e.g. cache-resident sizes) are repeated inside each
// sample until it lasts >= min_sample_ms, and the result is the time per call. This removes
// timer/thread wake-up noise at tiny sizes and measures warm steady state. Slow kernels
// (large N) run once per sample, exactly as before.
template <class F>
double bench_ms(F f, int reps = 7, double min_sample_ms = 5.0) {
    using clk = std::chrono::steady_clock;
    f();  // Warmup
    auto t0 = clk::now();
    f();
    double one = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    size_t inner = 1;
    if (one < min_sample_ms)
        inner = std::min<size_t>(100000, (size_t)std::ceil(min_sample_ms / std::max(one, 1e-4)));

    std::vector<double> t;
    for (int r = 0; r < reps; r++) {
        t0 = clk::now();
        for (size_t i = 0; i < inner; i++) f();
        double ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
        t.push_back(ms / inner);
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

static volatile float g_sink;

// work = bytes (-> GB/s) or FLOPs (-> GFLOP/s); both are "units per ms * 1e6"
static void row(const char* name, double ms, double base_ms, double work, const char* unit,
                double err) {
    printf("  %-16s %9.3f ms  %8.2f %-8s x%-6.2f  err=%.1e\n", name, ms, work / (ms * 1e6), unit,
           base_ms / ms, err);
}

int main(int argc, char** argv) {
    size_t n = argc > 1 ? strtoull(argv[1], nullptr, 10) : (1u << 24);  // 16M floats = 64 MB
    size_t m = argc > 2 ? strtoull(argv[2], nullptr, 10) : 768;
    m = (m + 15) / 16 * 16;  // Round GEMM size up to a multiple of 16

    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
#ifdef __ARM_NEON
    const char* neon = "active";
#else
    const char* neon = "NOT active (not aarch64?) - Scalar/Auto only";
#endif
    printf("OpenMP-Threads=%d, NEON: %s\n", threads, neon);
    printf("Timing: median of 7 samples; samples < 5 ms are averaged over repeated calls\n\n");

    // ---------------- Dot ----------------
    {
        std::vector<float> a(n), b(n);
        for (size_t i = 0; i < n; i++) {
            a[i] = 0.5f + 0.001f * (i % 1000);
            b[i] = 1.0f - 0.0005f * (i % 1000);
        }
        double flops = 2.0 * n;  // 1 FMA per element = 2 FLOP
        printf("Dot product, N=%zu floats (%.0f MB/array) - memory-bound, 0.25 FLOP/byte\n", n,
               n * 4 / 1e6);

        double ref = dot_ref(a.data(), b.data(), n);
        auto relerr = [&](float v) { return std::fabs(v - ref) / std::fabs(ref); };
        float r = 0.f;
        double base = bench_ms([&] { r = dot_scalar(a.data(), b.data(), n); g_sink = r; });
        row("scalar", base, base, flops, "GFLOP/s", relerr(r));
        double t = bench_ms([&] { r = dot_auto(a.data(), b.data(), n); g_sink = r; });
        row("auto (omp simd)", t, base, flops, "GFLOP/s", relerr(r));
#ifdef __ARM_NEON
        t = bench_ms([&] { r = dot_neon(a.data(), b.data(), n); g_sink = r; });
        row("neon", t, base, flops, "GFLOP/s", relerr(r));
        t = bench_ms([&] { r = dot_neon_omp(a.data(), b.data(), n); g_sink = r; });
        row("neon + openmp", t, base, flops, "GFLOP/s", relerr(r));
#endif
    }

    // ---------------- GEMM ----------------
    {
        std::vector<float> A(m * m), B(m * m), C(m * m), Cref(m * m);
        for (size_t i = 0; i < m * m; i++) {
            A[i] = 0.5f + 0.001f * (i % 97);
            B[i] = 1.0f - 0.001f * (i % 89);
        }
        double flops = 2.0 * m * m * m;
        printf("\nGEMM, n=%zu (%.0f MFLOP, %.1f MB data) - compute-bound, up to %.0f FLOP/byte\n", m,
               flops / 1e6, 3.0 * m * m * 4 / 1e6, m / 6.0);

        auto maxerr = [&] {
            double e = 0;
            for (size_t i = 0; i < m * m; i++)
                e = std::max(e, (double)std::fabs(C[i] - Cref[i]) / std::fabs(Cref[i]));
            return e;
        };
        const int R = 5;
        double base = bench_ms([&] { gemm_scalar(A.data(), B.data(), Cref.data(), m); }, R);
        row("scalar", base, base, flops, "GFLOP/s", 0.0);
        double t = bench_ms([&] { gemm_auto(A.data(), B.data(), C.data(), m); }, R);
        row("auto (omp simd)", t, base, flops, "GFLOP/s", maxerr());
#ifdef __ARM_NEON
        t = bench_ms([&] { gemm_neon(A.data(), B.data(), C.data(), m); }, R);
        row("neon 8x8 block", t, base, flops, "GFLOP/s", maxerr());
        t = bench_ms([&] { gemm_neon_omp(A.data(), B.data(), C.data(), m); }, R);
        row("neon + openmp", t, base, flops, "GFLOP/s", maxerr());
#endif
    }
    return 0;
}