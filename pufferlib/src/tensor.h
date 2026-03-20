#ifndef PUFFERLIB_TENSOR_H
#define PUFFERLIB_TENSOR_H

#include <stdint.h>

#define PUF_MAX_DIMS 8

typedef struct {
    float* data;
    int64_t shape[PUF_MAX_DIMS];
} FloatTensor;

typedef struct {
    unsigned char* data;
    int64_t shape[PUF_MAX_DIMS];
} ByteTensor;

typedef struct {
    long* data;
    int64_t shape[PUF_MAX_DIMS];
} LongTensor;

typedef struct {
    int* data;
    int64_t shape[PUF_MAX_DIMS];
} IntTensor;

#ifdef __CUDACC__
#include <cuda_bf16.h>

#ifdef PRECISION_FLOAT
typedef float precision_t;
constexpr bool USE_BF16 = false;
constexpr int PRECISION_SIZE = 4;
#define to_float(x) (x)
#define from_float(x) (x)
#else
typedef __nv_bfloat16 precision_t;
constexpr bool USE_BF16 = true;
constexpr int PRECISION_SIZE = 2;
#define to_float(x) __bfloat162float(x)
#define from_float(x) __float2bfloat16(x)
#endif

typedef struct {
    precision_t* data;
    int64_t shape[PUF_MAX_DIMS];
} PrecisionTensor;
#endif // __CUDACC__

#ifdef __CUDACC__
  #define PUF_INLINE __host__ __device__ inline
#elif defined(__cplusplus)
  #define PUF_INLINE inline
#else
  #define PUF_INLINE static inline
#endif

PUF_INLINE int ndim(const int64_t* shape) {
    int n = 0; while (n < PUF_MAX_DIMS && shape[n] != 0) n++; return n;
}

PUF_INLINE int64_t numel(const int64_t* shape) {
    int64_t n = 1; for (int i = 0; i < PUF_MAX_DIMS && shape[i] != 0; i++) n *= shape[i]; return n;
}

// Product of all dims except the last two (1 if ndim <= 2)
PUF_INLINE int64_t batch_size(const int64_t* shape) {
    int n = ndim(shape);
    int64_t b = 1;
    for (int i = 0; i < n - 2; i++) b *= shape[i];
    return b;
}

#ifdef __cplusplus
inline const char* _puf_repr_impl(const char* name, const char* dtype,
        const int64_t* shape, int nd, int64_t ne, bool empty) {
    static thread_local char buf[256];
    if (empty) { snprintf(buf, sizeof(buf), "%s(empty)", name); return buf; }
    int pos = snprintf(buf, sizeof(buf), "%s(%s, [", name, dtype);
    for (int i = 0; i < nd && pos < (int)sizeof(buf) - 32; i++)
        pos += snprintf(buf + pos, sizeof(buf) - pos, "%s%lld", i ? ", " : "", (long long)shape[i]);
    snprintf(buf + pos, sizeof(buf) - pos, "], %lld elems)", (long long)ne);
    return buf;
}

#ifdef __CUDACC__
inline const char* puf_repr(const PrecisionTensor* t) {
    return _puf_repr_impl("PrecisionTensor", USE_BF16 ? "bf16" : "f32",
        t->shape, ndim(t->shape), numel(t->shape), !t->data);
}
#endif

inline const char* puf_repr(const FloatTensor* t) {
    return _puf_repr_impl("FloatTensor", "f32",
        t->shape, ndim(t->shape), numel(t->shape), !t->data);
}
#endif // __cplusplus

#endif // PUFFERLIB_TENSOR_H
