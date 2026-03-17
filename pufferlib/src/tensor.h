#ifndef PUFFERLIB_TENSOR_H
#define PUFFERLIB_TENSOR_H

#include <stdint.h>

#define PUF_MAX_DIMS 8

typedef struct {
    float* data;
    int64_t shape[PUF_MAX_DIMS];
} FloatTensor;

typedef struct {
    double* data;
    int64_t shape[PUF_MAX_DIMS];
} DoubleTensor;

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

#endif // PUFFERLIB_TENSOR_H
