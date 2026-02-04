use crate::selection_state::{SelectionPhase, SelectionState};

pub fn draw_cursor(data: &mut [u8], w: usize, h: usize, x: i32, y: i32) {
    let x = x.clamp(0, w as i32 - 1) as usize;
    let y = y.clamp(0, h as i32 - 1) as usize;
    let plane = &mut data[..w * h];

    // Горизонтальная линия
    for px in x.saturating_sub(25)..=(x + 25).min(w - 1) {
        if !(x.saturating_sub(5)..=x + 5).contains(&px) {
            plane[y * w + px] = 255;
        }
    }

    // Вертикальная линия
    for py in y.saturating_sub(25)..=(y + 25).min(h - 1) {
        if !(y.saturating_sub(5)..=y + 5).contains(&py) {
            plane[py * w + x] = 255;
        }
    }
}

pub fn draw_selection(data: &mut [u8], w: usize, h: usize, sel: &SelectionState) {
    if sel.phase != SelectionPhase::SelectingArea {
        return;
    }

    let x1 = sel.start_x.min(sel.cursor_x).max(0) as usize;
    let y1 = sel.start_y.min(sel.cursor_y).max(0) as usize;
    let x2 = (sel.start_x.max(sel.cursor_x) as usize).min(w - 1);
    let y2 = (sel.start_y.max(sel.cursor_y) as usize).min(h - 1);

    let plane = &mut data[..w * h];

    // Пунктирная рамка
    for x in x1..=x2 {
        if (x / 6) % 2 == 0 {
            plane[y1 * w + x] = 255;
            plane[y2 * w + x] = 255;
        }
    }
    for y in y1..=y2 {
        if (y / 6) % 2 == 0 {
            plane[y * w + x1] = 255;
            plane[y * w + x2] = 255;
        }
    }
}