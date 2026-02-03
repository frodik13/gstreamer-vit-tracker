#![allow(non_camel_case_types)]
#![allow(dead_code)]

use libc::{c_int, c_void};

pub const RK_FORMAT_YCbCr_420_SP: c_int = 0x0;
pub const RK_FORMAT_RGB_888: c_int = 0x2;

#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct rga_buffer_t {
    pub vir_addr: *mut c_void,
    pub phy_addr: *mut c_void,
    pub handle: c_int,
    pub fd: c_int,
    pub width: c_int,
    pub height: c_int,
    pub wstride: c_int,
    pub hstride: c_int,
    pub format: c_int,
    pub color_space_mode: c_int,
    pub global_alpha: c_int,
    pub rd_mode: c_int,
}

#[cfg(feature = "rga")]
unsafe extern "C" {
    pub fn imcvtcolor(
        src: rga_buffer_t,
        dst: rga_buffer_t,
        src_format: c_int,
        dst_format: c_int,
        mode: c_int,
    ) -> c_int;
}

pub fn is_available() -> bool {
    cfg!(feature = "rga")
}

pub fn info() -> &'static str {
    if cfg!(feature = "rga") {
        "RGA hardware acceleration enabled"
    } else {
        "RGA not available, using CPU"
    }
}

#[cfg(feature = "rga")]
pub fn convert_nv12_to_rgb(
    nv12_data: &[u8],
    rgb_data: &mut [u8],
    width: usize,
    height: usize,
) -> Result<(), String> {
    let src = rga_buffer_t {
        vir_addr: nv12_data.as_ptr() as *mut c_void,
        width: width as c_int,
        height: height as c_int,
        wstride: width as c_int,
        hstride: height as c_int,
        format: RK_FORMAT_YCbCr_420_SP,
        global_alpha: 0xff,
        ..Default::default()
    };

    let dst = rga_buffer_t {
        vir_addr: rgb_data.as_mut_ptr() as *mut c_void,
        width: width as c_int,
        height: height as c_int,
        wstride: width as c_int,
        hstride: height as c_int,
        format: RK_FORMAT_RGB_888,
        global_alpha: 0xff,
        ..Default::default()
    };

    let ret = unsafe { imcvtcolor(src, dst, RK_FORMAT_YCbCr_420_SP, RK_FORMAT_RGB_888, 0) };

    if ret == 0 {
        Ok(())
    } else {
        Err(format!("RGA error: {}", ret))
    }
}

#[cfg(not(feature = "rga"))]
pub fn convert_nv12_to_rgb(
    _nv12_data: &[u8],
    _rgb_data: &mut [u8],
    _width: usize,
    _height: usize,
) -> Result<(), String> {
    Err("RGA not available".to_string())
}