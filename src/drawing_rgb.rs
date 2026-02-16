// drawing_rgb.rs
use crate::selection_state::{SelectionState, SelectionPhase};

#[inline]
fn set_pixel_rgb(data: &mut [u8], w: usize, x: i32, y: i32, h: usize, luma: u8) {
    if x < 0 || y < 0 || x >= w as i32 || y >= h as i32 {
        return;
    }
    let offset = (y as usize * w + x as usize) * 3;
    if offset + 2 < data.len() {
        data[offset] = luma;
        data[offset + 1] = luma;
        data[offset + 2] = luma;
    }
}

#[inline]
fn set_pixel_rgb_color(data: &mut [u8], w: usize, x: i32, y: i32, h: usize, r: u8, g: u8, b: u8) {
    if x < 0 || y < 0 || x >= w as i32 || y >= h as i32 {
        return;
    }
    let offset = (y as usize * w + x as usize) * 3;
    if offset + 2 < data.len() {
        data[offset] = r;
        data[offset + 1] = g;
        data[offset + 2] = b;
    }
}

pub fn draw_background_rgb(data: &mut [u8], w: usize, h: usize, x: i32, y: i32, bw: i32, bh: i32, _dim: u8) {
    // for row in y..(y + bh).min(h as i32) {
    //     for col in x..(x + bw).min(w as i32) {
    //         if row < 0 || col < 0 { continue; }
    //         let offset = (row as usize * w + col as usize) * 3;
    //         if offset + 2 < data.len() {
    //             data[offset] = data[offset] / 3;
    //             data[offset + 1] = data[offset + 1] / 3;
    //             data[offset + 2] = data[offset + 2] / 3;
    //         }
    //     }
    // }
    let x_start = x.max(0) as usize;
    let x_end = ((x + bw) as usize).min(w);
    let y_start = y.max(0) as usize;
    let y_end = ((y + bh) as usize).min(h);
    let row_bytes = (x_end - x_start) * 3;

    for row in y_start..y_end {
        let offset = (row * w + x_start) * 3;
        // Заполняем тёмно-серым без чтения (memset)
        data[offset..offset + row_bytes].fill(30);
    }
}

pub fn draw_rect_rgb(data: &mut [u8], w: usize, h: usize, x: i32, y: i32, rw: i32, rh: i32, thickness: i32, r: u8, g: u8, b: u8) {
    for t in 0..thickness {
        for i in 0..rw {
            set_pixel_rgb_color(data, w, x + i, y + t, h, r, g, b);
            set_pixel_rgb_color(data, w, x + i, y + rh - 1 - t, h, r, g, b);
        }
        for i in 0..rh {
            set_pixel_rgb_color(data, w, x + t, y + i, h, r, g, b);
            set_pixel_rgb_color(data, w, x + rw - 1 - t, y + i, h, r, g, b);
        }
    }
}

pub fn draw_crosshair_rgb(data: &mut [u8], w: usize, h: usize, cx: i32, cy: i32, size: i32, r: u8, g: u8, b: u8) {
    for i in -size..=size {
        set_pixel_rgb_color(data, w, cx + i, cy, h, r, g, b);
        set_pixel_rgb_color(data, w, cx, cy + i, h, r, g, b);
    }
}

pub fn draw_cursor_rgb(data: &mut [u8], w: usize, h: usize, cx: i32, cy: i32) {
    let size = 25;
    let gap = 5;
    for i in gap..=size {
        set_pixel_rgb_color(data, w, cx + i, cy, h, 0, 255, 0);
        set_pixel_rgb_color(data, w, cx - i, cy, h, 0, 255, 0);
        set_pixel_rgb_color(data, w, cx, cy + i, h, 0, 255, 0);
        set_pixel_rgb_color(data, w, cx, cy - i, h, 0, 255, 0);
    }
}

pub fn draw_text_rgb(data: &mut [u8], w: usize, h: usize, text: &str, x: i32, y: i32, scale: i32, luma: u8) {
    let mut cx = x;
    for ch in text.chars() {
        if let Ok(glyph) = crate::drawing::get_glyph(ch) {
            for (gy, &bits) in glyph.iter().enumerate() {
                for gx in 0..5i32 {
                    if (bits >> (4 - gx)) & 1 == 1 {
                        for sy in 0..scale {
                            for sx in 0..scale {
                                set_pixel_rgb(data, w, cx + gx * scale + sx, y + gy as i32 * scale + sy, h, luma);
                            }
                        }
                    }
                }
            }
        }
        cx += 6 * scale;
    }
}

pub fn draw_selection_rgb(data: &mut [u8], w: usize, h: usize, sel: &SelectionState) {
    if sel.phase != SelectionPhase::SelectingArea {
        return;
    }

    let x1 = sel.start_x.min(sel.cursor_x).max(0);
    let y1 = sel.start_y.min(sel.cursor_y).max(0);
    let x2 = sel.start_x.max(sel.cursor_x).min(w as i32 - 1);
    let y2 = sel.start_y.max(sel.cursor_y).min(h as i32 - 1);

    // Пунктирная рамка
    for x in x1..=x2 {
        if (x / 6) % 2 == 0 {
            set_pixel_rgb_color(data, w, x, y1, h, 255, 255, 0);
            set_pixel_rgb_color(data, w, x, y2, h, 255, 255, 0);
        }
    }
    for y in y1..=y2 {
        if (y / 6) % 2 == 0 {
            set_pixel_rgb_color(data, w, x1, y, h, 255, 255, 0);
            set_pixel_rgb_color(data, w, x2, y, h, 255, 255, 0);
        }
    }
}