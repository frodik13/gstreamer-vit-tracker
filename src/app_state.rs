#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AppState {
    Selecting,
    Tracking,
    Lost { frames: u64 },
}