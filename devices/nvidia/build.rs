fn main() {
    use build_script_cfg::Cfg;
    use search_cuda_tools::find_cuda_root;

    let cuda = Cfg::new("detected_cuda");
    if find_cuda_root().is_some() {
        cuda.define();
        println!("cargo:rerun-if-changed=src/sample.cu");
        cc::Build::new()
            .cuda(true)
            .flag("-gencode")
            .flag("arch=compute_80,code=sm_80")
            .flag("-std=c++20")
            .flag("-allow-unsupported-compiler")
            .file("src/sample.cu")
            .compile("sample");
    }
}
