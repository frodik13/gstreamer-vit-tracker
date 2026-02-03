fn main() {
    // Пробуем найти librga
    if pkg_config::probe_library("librga").is_ok() {
        println!("cargo:rustc-cfg=feature=\"rga\"");
    } else if std::path::Path::new("/usr/lib/aarch64-linux-gnu/librga.so").exists() {
        println!("cargo:rustc-cfg=feature=\"rga\"");
        println!("cargo:rustc-link-search=/usr/lib/aarch64-linux-gnu");
        println!("cargo:rustc-link-lib=rga");
    } else {
        println!("cargo:warning=librga not found, using CPU conversion");
    }
}