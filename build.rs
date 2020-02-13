fn main() {
    #[cfg(feature = "blas")]
    {
        println!("cargo:rustc-link-lib=dylib=blas");
    }
}
