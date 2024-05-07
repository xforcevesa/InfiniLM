#include <cub/device/device_reduce.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cuda_fp16.h>

extern "C" cudaError argmax_half(
    void *temp_storage, size_t *temp_storage_bytes,
    half const *input, int num_items,
    cub::KeyValuePair<int, half> *output,
    cudaStream_t stream)
{
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
    cudaStream_t stream)
{
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
    cudaStream_t stream)
{
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
        if (error != cudaSuccess)                                                               \
        {                                                                                       \
            printf("Error: %s (%d) at \"%s\"\n", cudaGetErrorString(error), error, #statement); \
            return error;                                                                       \
        }                                                                                       \
    }

static __global__ void partial_softmax_half_kernel(
    half *__restrict__ data,
    half const *__restrict__ max_,
    float temperature,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        data[i] = hexp((data[i] - __ldg(max_)) / (half)temperature);
    }
}

extern "C" cudaError partial_softmax_half(
    half *data,
    float temperature,
    int topk,
    cudaStream_t stream)
{
    half *max_ = nullptr;
    RUNTIME(cudaMallocAsync(&max_, sizeof(half), stream))
    RUNTIME(cudaMemcpyAsync(max_, data, sizeof(half), cudaMemcpyDeviceToDevice, stream))

    auto block = min(1024, topk);
    auto grid = (topk + block - 1) / block;
    partial_softmax_half_kernel<<<grid, block, 0, stream>>>(data, max_, temperature, topk);

    RUNTIME(cudaFreeAsync(max_, stream))
    return cudaGetLastError();
}

static __global__ void random_sample_kernel(
    half const *__restrict__ data,
    unsigned int const *__restrict__ indices,
    unsigned int *__restrict__ index_,
    float probability,
    int n)
{
    half p = probability * (float)data[n - 1];
    for (int i = 0; i < n; ++i)
    {
        if (data[i] >= p)
        {
            *index_ = indices[i];
            return;
        }
    }
}

extern "C" cudaError random_sample_half(
    half const *data,
    unsigned int const *indices,
    unsigned int *index,
    float probability,
    int topk,
    cudaStream_t stream)
{
    unsigned int *index_ = nullptr;
    cudaMallocAsync(&index_, sizeof(unsigned int), stream);

    random_sample_kernel<<<1, 1, 0, stream>>>(data, indices, index_, probability, topk);

    cudaMemcpy(index, index_, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaFree(index_);

    return cudaGetLastError();
}
