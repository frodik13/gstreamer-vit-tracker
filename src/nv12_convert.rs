use ndarray::Array3;
use rayon::prelude::*;
use std::sync::OnceLock;

// Предвычисленные таблицы для быстрой конвертации
static YUV_TABLES: OnceLock<YuvTables> = OnceLock::new();

struct YuvTables {
    y_table: [i32; 256],    // 298 * (y - 16)
    rv_table: [i32; 256],   // 409 * (v - 128)
    gu_table: [i32; 256],   // 100 * (u - 128)
    gv_table: [i32; 256],   // 208 * (v - 128)
    bu_table: [i32; 256],   // 516 * (u - 128)
}

impl YuvTables {
    fn new() -> Self {
        let mut y_table = [0i32; 256];
        let mut rv_table = [0i32; 256];
        let mut gu_table = [0i32; 256];
        let mut gv_table = [0i32; 256];
        let mut bu_table = [0i32; 256];

        for i in 0..256 {
            y_table[i] = 298 * (i as i32 - 16);
            rv_table[i] = 409 * (i as i32 - 128);
            gu_table[i] = 100 * (i as i32 - 128);
            gv_table[i] = 208 * (i as i32 - 128);
            bu_table[i] = 516 * (i as i32 - 128);
        }

        Self { y_table, rv_table, gu_table, gv_table, bu_table }
    }
}

fn get_tables() -> &'static YuvTables {
    YUV_TABLES.get_or_init(YuvTables::new)
}

#[inline(always)]
fn clamp_u8(val: i32) -> u8 {
    if val < 0 { 0 } else if val > 255 { 255 } else { val as u8 }
}

/// Максимально оптимизированная конвертация NV12 -> BGR
pub fn nv12_full_to_bgr_parallel(nv12_data: &[u8], width: usize, height: usize) -> Array3<u8> {
    let y_plane_size = width * height;
    if nv12_data.len() < y_plane_size * 3 / 2 {
        return Array3::<u8>::zeros((height, width, 3));
    }

    let tables = get_tables();
    let y_plane = &nv12_data[..y_plane_size];
    let uv_plane = &nv12_data[y_plane_size..];

    let mut output = vec![0u8; height * width * 3];

    // Обрабатываем по 2 строки за раз (они делят одну UV строку)
    output
        .par_chunks_mut(width * 3 * 2)
        .enumerate()
        .for_each(|(pair_idx, rows_data)| {
            let row0 = pair_idx * 2;
            let row1 = row0 + 1;
            let uv_row = pair_idx;

            if row1 >= height {
                // Последняя нечётная строка
                if row0 < height {
                    process_row_unsafe(
                        y_plane,
                        uv_plane,
                        &mut rows_data[..width * 3],
                        row0,
                        uv_row,
                        width,
                        tables,
                    );
                }
                return;
            }

            let (row0_data, row1_data) = rows_data.split_at_mut(width * 3);

            // Обрабатываем обе строки
            process_row_unsafe(y_plane, uv_plane, row0_data, row0, uv_row, width, tables);
            process_row_unsafe(y_plane, uv_plane, row1_data, row1, uv_row, width, tables);
        });

    Array3::from_shape_vec((height, width, 3), output)
        .unwrap_or_else(|_| Array3::zeros((height, width, 3)))
}

#[inline(always)]
fn process_row_unsafe(
    y_plane: &[u8],
    uv_plane: &[u8],
    row_data: &mut [u8],
    row: usize,
    uv_row: usize,
    width: usize,
    tables: &YuvTables,
) {
    let y_row_start = row * width;
    let uv_row_start = uv_row * width;

    // Обрабатываем по 2 пикселя (они делят одну UV пару)
    let mut col = 0;
    while col + 1 < width {
        unsafe {
            let uv_idx = uv_row_start + col;
            let u = *uv_plane.get_unchecked(uv_idx) as usize;
            let v = *uv_plane.get_unchecked(uv_idx + 1) as usize;

            let rv = tables.rv_table[v];
            let gu = tables.gu_table[u];
            let gv = tables.gv_table[v];
            let bu = tables.bu_table[u];

            // Первый пиксель
            let y0 = *y_plane.get_unchecked(y_row_start + col) as usize;
            let y_val0 = tables.y_table[y0];

            let r0 = (y_val0 + rv + 128) >> 8;
            let g0 = (y_val0 - gu - gv + 128) >> 8;
            let b0 = (y_val0 + bu + 128) >> 8;

            let out_idx0 = col * 3;
            *row_data.get_unchecked_mut(out_idx0) = clamp_u8(b0);
            *row_data.get_unchecked_mut(out_idx0 + 1) = clamp_u8(g0);
            *row_data.get_unchecked_mut(out_idx0 + 2) = clamp_u8(r0);

            // Второй пиксель
            let y1 = *y_plane.get_unchecked(y_row_start + col + 1) as usize;
            let y_val1 = tables.y_table[y1];

            let r1 = (y_val1 + rv + 128) >> 8;
            let g1 = (y_val1 - gu - gv + 128) >> 8;
            let b1 = (y_val1 + bu + 128) >> 8;

            let out_idx1 = (col + 1) * 3;
            *row_data.get_unchecked_mut(out_idx1) = clamp_u8(b1);
            *row_data.get_unchecked_mut(out_idx1 + 1) = clamp_u8(g1);
            *row_data.get_unchecked_mut(out_idx1 + 2) = clamp_u8(r1);
        }
        col += 2;
    }

    // Последний пиксель если ширина нечётная
    if col < width {
        unsafe {
            let uv_idx = uv_row_start + (col / 2) * 2;
            let u = *uv_plane.get_unchecked(uv_idx) as usize;
            let v = *uv_plane.get_unchecked(uv_idx + 1) as usize;

            let y0 = *y_plane.get_unchecked(y_row_start + col) as usize;
            let y_val0 = tables.y_table[y0];

            let r = (y_val0 + tables.rv_table[v] + 128) >> 8;
            let g = (y_val0 - tables.gu_table[u] - tables.gv_table[v] + 128) >> 8;
            let b = (y_val0 + tables.bu_table[u] + 128) >> 8;

            let out_idx = col * 3;
            *row_data.get_unchecked_mut(out_idx) = clamp_u8(b);
            *row_data.get_unchecked_mut(out_idx + 1) = clamp_u8(g);
            *row_data.get_unchecked_mut(out_idx + 2) = clamp_u8(r);
        }
    }
}

/// Рисует прямоугольник
pub fn draw_rect_nv12(
    nv12_data: &mut [u8],
    width: usize,
    height: usize,
    x: i32,
    y: i32,
    w: i32,
    h: i32,
    thickness: usize,
    brightness: u8,
) {
    let x1 = x.max(0) as usize;
    let y1 = y.max(0) as usize;
    let x2 = ((x + w) as usize).min(width.saturating_sub(1));
    let y2 = ((y + h) as usize).min(height.saturating_sub(1));

    let y_plane = &mut nv12_data[..width * height];

    for t in 0..thickness {
        if y1 + t < height {
            for px in x1..=x2 {
                y_plane[(y1 + t) * width + px] = brightness;
            }
        }
        if y2 >= t && y2 - t < height {
            for px in x1..=x2 {
                y_plane[(y2 - t) * width + px] = brightness;
            }
        }
    }

    for py in y1..=y2 {
        for t in 0..thickness {
            if x1 + t < width {
                y_plane[py * width + x1 + t] = brightness;
            }
            if x2 >= t && x2 - t < width {
                y_plane[py * width + x2 - t] = brightness;
            }
        }
    }
}

/// Рисует перекрестие
pub fn draw_crosshair_nv12(
    nv12_data: &mut [u8],
    width: usize,
    height: usize,
    cx: i32,
    cy: i32,
    size: i32,
    brightness: u8,
) {
    let cx = cx.max(0) as usize;
    let cy = cy.max(0) as usize;
    let size = size as usize;

    let y_plane = &mut nv12_data[..width * height];

    if cy < height {
        for x in cx.saturating_sub(size)..=(cx + size).min(width - 1) {
            y_plane[cy * width + x] = brightness;
        }
    }

    if cx < width {
        for y in cy.saturating_sub(size)..=(cy + size).min(height - 1) {
            y_plane[y * width + cx] = brightness;
        }
    }
}

/// Рисует текст
pub fn draw_text_nv12(
    nv12_data: &mut [u8],
    width: usize,
    height: usize,
    text: &str,
    x: usize,
    y: usize,
    scale: usize,
    brightness: u8,
) {
    const FONT: &[(&str, [u8; 7])] = &[
        ("0", [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110]),
        ("1", [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110]),
        ("2", [0b01110, 0b10001, 0b00001, 0b00110, 0b01000, 0b10000, 0b11111]),
        ("3", [0b01110, 0b10001, 0b00001, 0b00110, 0b00001, 0b10001, 0b01110]),
        ("4", [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010]),
        ("5", [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110]),
        ("6", [0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110]),
        ("7", [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000]),
        ("8", [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110]),
        ("9", [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100]),
        (".", [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b01100, 0b01100]),
        (":", [0b00000, 0b01100, 0b01100, 0b00000, 0b01100, 0b01100, 0b00000]),
        ("-", [0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000]),
        (" ", [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000]),
        ("F", [0b11111, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000, 0b10000]),
        ("P", [0b11110, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000, 0b10000]),
        ("S", [0b01110, 0b10001, 0b10000, 0b01110, 0b00001, 0b10001, 0b01110]),
        ("T", [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100]),
        ("R", [0b11110, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001, 0b10001]),
        ("A", [0b01110, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001, 0b10001]),
        ("C", [0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110]),
        ("K", [0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001]),
        ("I", [0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110]),
        ("N", [0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001]),
        ("G", [0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01110]),
        ("E", [0b11111, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000, 0b11111]),
        ("L", [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111]),
        ("O", [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110]),
        ("D", [0b11100, 0b10010, 0b10001, 0b10001, 0b10001, 0b10010, 0b11100]),
        ("%", [0b11001, 0b11010, 0b00100, 0b00100, 0b01000, 0b01011, 0b10011]),
        ("s", [0b00000, 0b00000, 0b01110, 0b10000, 0b01110, 0b00001, 0b11110]),
        ("c", [0b00000, 0b00000, 0b01110, 0b10000, 0b10000, 0b10001, 0b01110]),
        ("o", [0b00000, 0b00000, 0b01110, 0b10001, 0b10001, 0b10001, 0b01110]),
        ("r", [0b00000, 0b00000, 0b10110, 0b11001, 0b10000, 0b10000, 0b10000]),
        ("e", [0b00000, 0b00000, 0b01110, 0b10001, 0b11111, 0b10000, 0b01110]),
        ("m", [0b00000, 0b00000, 0b11010, 0b10101, 0b10101, 0b10001, 0b10001]),
        ("t", [0b01000, 0b01000, 0b11100, 0b01000, 0b01000, 0b01001, 0b00110]),
        ("k", [0b10000, 0b10000, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010]),
        ("n", [0b00000, 0b00000, 0b10110, 0b11001, 0b10001, 0b10001, 0b10001]),
        ("v", [0b00000, 0b00000, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100]),
    ];

    let y_plane = &mut nv12_data[..width * height];
    let mut cursor_x = x;

    for ch in text.chars() {
        if let Some((_, glyph)) = FONT.iter().find(|(c, _)| c.chars().next() == Some(ch)) {
            for (row, &bits) in glyph.iter().enumerate() {
                for col in 0..5 {
                    if (bits >> (4 - col)) & 1 == 1 {
                        for dy in 0..scale {
                            for dx in 0..scale {
                                let px = cursor_x + col * scale + dx;
                                let py = y + row * scale + dy;
                                if px < width && py < height {
                                    y_plane[py * width + px] = brightness;
                                }
                            }
                        }
                    }
                }
            }
        }
        cursor_x += 6 * scale;
    }
}

/// Затемняет область
pub fn draw_background_nv12(
    nv12_data: &mut [u8],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    darkness: u8,
) {
    let y_plane = &mut nv12_data[..width * height];
    let factor = (255 - darkness) as u16;

    for py in y..(y + h).min(height) {
        for px in x..(x + w).min(width) {
            let idx = py * width + px;
            y_plane[idx] = ((y_plane[idx] as u16 * factor) / 255) as u8;
        }
    }
}