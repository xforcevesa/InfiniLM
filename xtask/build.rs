fn main() {
    use search_cuda_tools::{allow_cfg, detect, find_cuda_root, find_nccl_root};

    allow_cfg("cuda");
    allow_cfg("nccl");
    if !cfg!(feature = "nvidia") {
        return;
    }
    if find_cuda_root().is_some() {
        detect("cuda");
    }
    if find_nccl_root().is_some() {
        detect("nccl");
    }
}
