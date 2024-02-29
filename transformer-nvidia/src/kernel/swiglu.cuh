static __forceinline__ __device__ float sigmoid(float x) {
    return fdividef(1, 1 + expf(-x));
}

template<class Tdata>
static __device__ void swiglu(
    Tdata *__restrict__ gate_,
    int const stride_gate,
    Tdata const *__restrict__ up_,
    int const stride_up) {
    auto i = blockIdx.y * stride_gate + blockIdx.x * blockDim.x + threadIdx.x,
         j = blockIdx.y * stride_up + blockIdx.x * blockDim.x + threadIdx.x;
    auto x = float(gate_[i]),
         y = float(up_[j]);
    gate_[i] = Tdata(x * sigmoid(x) * y);
}
