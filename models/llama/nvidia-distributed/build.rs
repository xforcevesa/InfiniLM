fn main() {
    use search_cuda_tools::{allow_cfg, detect, find_nccl_root};

    allow_cfg("nccl");
    if find_nccl_root().is_some() {
        detect("nccl");
    }
}
