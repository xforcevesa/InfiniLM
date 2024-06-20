fn main() {
    use build_script_cfg::Cfg;
    use search_cuda_tools::find_nccl_root;

    let nccl = Cfg::new("detected_nccl");
    if find_nccl_root().is_some() {
        nccl.define();
    }
}
