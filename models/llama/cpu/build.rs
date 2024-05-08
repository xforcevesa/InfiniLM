fn main() {
    if !intel_mkl_tool::Library::available().is_empty() {
        // **NOTICE** add mkl_rt to library search path
        println!("cargo:rustc-cfg=detected_mkl");
        println!("cargo:rustc-link-lib=dylib=mkl_rt");
    }
}
