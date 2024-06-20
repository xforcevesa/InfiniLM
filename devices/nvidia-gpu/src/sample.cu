#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_scan.cuh>
#include <cuda_fp16.h>

extern "C" cudaError argmax_half(
    void *temp_storage, size_t *temp_storage_bytes,
    half const *input, int num_items,
    cub::KeyValuePair<int, half> *output,
    cudaStream_t stream) {
    return cub::DeviceReduce::ArgMax(
        temp_storage, *temp_storage_bytes,
        input,
        output,
        num_items,
        stream);
}

extern "C" cudaError radix_sort_half(
    void *temp_storage, size_t *temp_storage_bytes,
    half const *key_in, half *key_out,
    unsigned int const *value_in, unsigned int *value_out,
    int num_items,
    cudaStream_t stream) {
    return cub::DeviceRadixSort::SortPairsDescending(
        temp_storage, *temp_storage_bytes,
        key_in,
        key_out,
        value_in,
        value_out,
        num_items,
        0,
        sizeof(half) * 8,
        stream);
}

extern "C" cudaError inclusive_sum_half(
    void *temp_storage, size_t *temp_storage_bytes,
    half *data, int num_items,
    cudaStream_t stream) {
    return cub::DeviceScan::InclusiveSum(
        temp_storage, *temp_storage_bytes,
        data,
        data,
        num_items,
        stream);
}

#define RUNTIME(statement)                                                                      \
    {                                                                                           \
        auto error = statement;                                                                 \
        if (error != cudaSuccess) {                                                             \
            printf("Error: %s (%d) at \"%s\"\n", cudaGetErrorString(error), error, #statement); \
            return error;                                                                       \
        }                                                                                       \
    }

static __global__ void partial_softmax_half_kernel(
    half2 *__restrict__ data,
    float temperature,
    int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (0 < i && i < n) {
        auto max = __ldg((half *) data);
        data[i] = h2exp((data[i] - half2(max, max)) / half2(temperature, temperature));
    }
}

static __global__ void set_softmax_max_kernel(
    half *__restrict__ data, float temperature) {
    data[1] = hexp((data[1] - data[0]) / (half) temperature);
    data[0] = 1;
}

extern "C" cudaError partial_softmax_half(
    half *data,
    float temperature,
    int voc,
    cudaStream_t stream) {

    voc /= 2;
    auto block = min(1024, voc);
    auto grid = (voc + block - 1) / block;
    partial_softmax_half_kernel<<<grid, block, 0, stream>>>((half2 *) data, temperature, voc);
    set_softmax_max_kernel<<<1, 1, 0, stream>>>(data, temperature);

    return cudaGetLastError();
}

static __global__ void random_sample_kernel(
    half const *__restrict__ data,
    unsigned int const *__restrict__ indices,
    unsigned int *__restrict__ index_,
    float random, float topp, int topk, int voc) {
    half p = random * min(topp * (float) data[voc - 1], (float) data[topk - 1]);
    for (int i = 0;; ++i) {
        if (data[i] >= p) {
            *index_ = indices[i];
            return;
        }
    }
}

extern "C" cudaError random_sample_half(
    half const *data,
    unsigned int const *indices,
    unsigned int *index,
    float random, float topp, int topk, int voc,
    cudaStream_t stream) {
    unsigned int *index_ = nullptr;
    cudaMallocAsync(&index_, sizeof(unsigned int), stream);

    random_sample_kernel<<<1, 1, 0, stream>>>(data, indices, index_, random, topp, topk, voc);

    cudaMemcpy(index, index_, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaFree(index_);

    return cudaGetLastError();
}
