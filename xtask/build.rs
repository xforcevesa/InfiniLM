fn main() {
    use build_script_cfg::Cfg;
    use search_cuda_tools::{find_cuda_root, find_nccl_root};

    let cuda = Cfg::new("detected_cuda");
    let nccl = Cfg::new("detected_nccl");
    if !cfg!(feature = "nvidia") {
        return;
    }
    if find_cuda_root().is_some() {
        cuda.define();
    }
    if find_nccl_root().is_some() {
        nccl.define();
    }
}
