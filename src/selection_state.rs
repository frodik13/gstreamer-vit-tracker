use vit_tracker::BBox;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SelectionPhase {
    MovingToStart,
    SelectingArea,
}

#[derive(Debug, Clone)]
pub struct SelectionState {
    pub(crate) cursor_x: i32,
    pub(crate) cursor_y: i32,
    pub(crate) start_x: i32,
    pub(crate) start_y: i32,
    pub(crate) phase: SelectionPhase,
    pub(crate) step: i32,
    pub(crate) fast_step: i32,
}

impl SelectionState {
    pub fn new(width: i32, height: i32) -> Self {
        Self {
            cursor_x: width / 2,
            cursor_y: height / 2,
            start_x: width / 2,
            start_y: height / 2,
            phase: SelectionPhase::MovingToStart,
            step: 10,
            fast_step: 50,
        }
    }

    pub fn move_cursor(&mut self, dx: i32, dy: i32, fast: bool, width: i32, height: i32) {
        let step = if fast { self.fast_step } else { self.step };
        self.cursor_x = (self.cursor_x + dx * step).clamp(0, width - 1);
        self.cursor_y = (self.cursor_y + dy * step).clamp(0, height - 1);
    }

    pub fn get_bbox(&self) -> BBox {
        let x = self.start_x.min(self.cursor_x);
        let y = self.start_y.min(self.cursor_y);
        let w = (self.start_x - self.cursor_x).abs().max(20);
        let h = (self.start_y - self.cursor_y).abs().max(20);
        BBox::new(x, y, w, h)
    }
}