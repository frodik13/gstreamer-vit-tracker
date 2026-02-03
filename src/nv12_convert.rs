use ndarray::Array3;

/// Порядок каналов
#[derive(Debug, Clone, Copy)]
pub enum ColorOrder {
    RGB,
    BGR,
}

/// Конвертирует ROI из NV12 в RGB или BGR
pub fn nv12_roi_to_array(
    nv12_data: &[u8],
    frame_width: usize,
    frame_height: usize,
    roi_x: usize,
    roi_y: usize,
    roi_width: usize,
    roi_height: usize,
    color_order: ColorOrder,
) -> Array3<u8> {
    let x = roi_x.min(frame_width.saturating_sub(1));
    let y = roi_y.min(frame_height.saturating_sub(1));
    let w = roi_width.min(frame_width - x);
    let h = roi_height.min(frame_height - y);

    let mut result = Array3::<u8>::zeros((h, w, 3));

    let y_plane_size = frame_width * frame_height;
    if nv12_data.len() < y_plane_size * 3 / 2 {
        return result;
    }

    let y_plane = &nv12_data[..y_plane_size];
    let uv_plane = &nv12_data[y_plane_size..];

    for row in 0..h {
        for col in 0..w {
            let src_x = x + col;
            let src_y = y + row;

            let y_val = y_plane[src_y * frame_width + src_x] as i32;

            let uv_x = src_x / 2;
            let uv_y = src_y / 2;
            let uv_idx = uv_y * frame_width + uv_x * 2;

            // NV12: UV interleaved (U first, then V)
            let u = uv_plane.get(uv_idx).copied().unwrap_or(128) as i32 - 128;
            let v = uv_plane.get(uv_idx + 1).copied().unwrap_or(128) as i32 - 128;

            // YUV to RGB (BT.601)
            let c = y_val - 16;
            let r = ((298 * c + 409 * v + 128) >> 8).clamp(0, 255) as u8;
            let g = ((298 * c - 100 * u - 208 * v + 128) >> 8).clamp(0, 255) as u8;
            let b = ((298 * c + 516 * u + 128) >> 8).clamp(0, 255) as u8;

            match color_order {
                ColorOrder::RGB => {
                    result[[row, col, 0]] = r;
                    result[[row, col, 1]] = g;
                    result[[row, col, 2]] = b;
                }
                ColorOrder::BGR => {
                    result[[row, col, 0]] = b;
                    result[[row, col, 1]] = g;
                    result[[row, col, 2]] = r;
                }
            }
        }
    }

    result
}

/// Конвертирует ПОЛНЫЙ кадр NV12 в BGR (как OpenCV Mat)
/// Используйте это если трекер теряет объект с ROI
pub fn nv12_full_to_bgr(
    nv12_data: &[u8],
    width: usize,
    height: usize,
) -> Array3<u8> {
    nv12_roi_to_array(nv12_data, width, height, 0, 0, width, height, ColorOrder::BGR)
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
        ("W", [0b10001, 0b10001, 0b10101, 0b10101, 0b10101, 0b11011, 0b10001]),
        ("L", [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111]),
        ("O", [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110]),
        ("E", [0b11111, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000, 0b11111]),
        ("D", [0b11100, 0b10010, 0b10001, 0b10001, 0b10001, 0b10010, 0b11100]),
        ("B", [0b11110, 0b10001, 0b11110, 0b10001, 0b10001, 0b10001, 0b11110]),
        ("m", [0b00000, 0b00000, 0b11010, 0b10101, 0b10101, 0b10001, 0b10001]),
        ("s", [0b00000, 0b00000, 0b01110, 0b10000, 0b01110, 0b00001, 0b11110]),
        ("%", [0b11001, 0b11010, 0b00100, 0b00100, 0b01000, 0b01011, 0b10011]),
        ("|", [0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100]),
        ("x", [0b00000, 0b00000, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001]),
        ("v", [0b00000, 0b00000, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100]),
        ("a", [0b00000, 0b00000, 0b01110, 0b00001, 0b01111, 0b10001, 0b01111]),
        ("r", [0b00000, 0b00000, 0b10110, 0b11001, 0b10000, 0b10000, 0b10000]),
        ("c", [0b00000, 0b00000, 0b01110, 0b10000, 0b10000, 0b10001, 0b01110]),
        ("k", [0b10000, 0b10000, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010]),
        ("t", [0b01000, 0b01000, 0b11100, 0b01000, 0b01000, 0b01001, 0b00110]),
        ("o", [0b00000, 0b00000, 0b01110, 0b10001, 0b10001, 0b10001, 0b01110]),
        ("p", [0b00000, 0b00000, 0b11110, 0b10001, 0b11110, 0b10000, 0b10000]),
        ("y", [0b00000, 0b00000, 0b10001, 0b10001, 0b01111, 0b00001, 0b01110]),
        ("n", [0b00000, 0b00000, 0b10110, 0b11001, 0b10001, 0b10001, 0b10001]),
        ("i", [0b00100, 0b00000, 0b01100, 0b00100, 0b00100, 0b00100, 0b01110]),
        ("g", [0b00000, 0b00000, 0b01111, 0b10001, 0b01111, 0b00001, 0b01110]),
        ("l", [0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110]),
        ("e", [0b00000, 0b00000, 0b01110, 0b10001, 0b11111, 0b10000, 0b01110]),
        ("d", [0b00001, 0b00001, 0b01111, 0b10001, 0b10001, 0b10001, 0b01111]),
        ("f", [0b00110, 0b01001, 0b01000, 0b11100, 0b01000, 0b01000, 0b01000]),
        ("w", [0b00000, 0b00000, 0b10001, 0b10001, 0b10101, 0b10101, 0b01010]),
        ("[", [0b01110, 0b01000, 0b01000, 0b01000, 0b01000, 0b01000, 0b01110]),
        ("]", [0b01110, 0b00010, 0b00010, 0b00010, 0b00010, 0b00010, 0b01110]),
        ("u", [0b00000, 0b00000, 0b10001, 0b10001, 0b10001, 0b10011, 0b01101]),
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