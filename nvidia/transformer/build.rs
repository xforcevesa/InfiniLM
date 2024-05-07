fn main() {
    if search_cuda_tools::find_cuda_root().is_some() {
        search_cuda_tools::detect_cuda();
        println!("cargo:rerun-if-changed=src/sample.cu");
        cc::Build::new()
            .cuda(true)
            .flag("-gencode")
            .flag("arch=compute_80,code=sm_80")
            .file("src/sample.cu")
            .compile("sample");
    }
}
