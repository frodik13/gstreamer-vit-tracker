use ndarray::Array3;
use rayon::prelude::*;

/// Параллельная конвертация полного кадра NV12 -> BGR
pub fn nv12_full_to_bgr_parallel(nv12_data: &[u8], width: usize, height: usize) -> Array3<u8> {
    let y_plane_size = width * height;
    if nv12_data.len() < y_plane_size * 3 / 2 {
        return Array3::<u8>::zeros((height, width, 3));
    }

    let y_plane = &nv12_data[..y_plane_size];
    let uv_plane = &nv12_data[y_plane_size..];

    let mut output = vec![0u8; height * width * 3];

    // Параллельная обработка по строкам
    output
        .par_chunks_mut(width * 3)
        .enumerate()
        .for_each(|(row, row_data)| {
            let uv_row = row / 2;

            for col in 0..width {
                let y_val = y_plane[row * width + col] as i32;

                let uv_col = col / 2;
                let uv_idx = uv_row * width + uv_col * 2;

                let u = uv_plane.get(uv_idx).copied().unwrap_or(128) as i32 - 128;
                let v = uv_plane.get(uv_idx + 1).copied().unwrap_or(128) as i32 - 128;

                let c = y_val - 16;
                let r = ((298 * c + 409 * v + 128) >> 8).clamp(0, 255) as u8;
                let g = ((298 * c - 100 * u - 208 * v + 128) >> 8).clamp(0, 255) as u8;
                let b = ((298 * c + 516 * u + 128) >> 8).clamp(0, 255) as u8;

                let idx = col * 3;
                row_data[idx] = b;
                row_data[idx + 1] = g;
                row_data[idx + 2] = r;
            }
        });

    Array3::from_shape_vec((height, width, 3), output).unwrap_or_else(|_| Array3::zeros((height, width, 3)))
}

/// Рисует прямоугольник на Y-плоскости NV12
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