fn main() {
    use search_cuda_tools::*;
    if !cfg!(feature = "nvidia") {
        return;
    }
    if find_cuda_root().is_some() {
        detect_cuda();
    }
    if find_nccl_root().is_some() {
        detect_nccl();
    }
}
