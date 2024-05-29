fn main() {
    use search_cuda_tools::{allow_cfg, detect, find_cuda_root};

    allow_cfg("cuda");
    if find_cuda_root().is_some() {
        detect("cuda");
        println!("cargo:rerun-if-changed=src/sample.cu");
        cc::Build::new()
            .cuda(true)
            .flag("-gencode")
            .flag("arch=compute_80,code=sm_80")
            .flag("-allow-unsupported-compiler")
            .file("src/sample.cu")
            .compile("sample");
    }
}
