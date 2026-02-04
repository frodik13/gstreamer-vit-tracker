#[derive(Debug, Clone)]
pub enum UserCommand {
    MoveUp(bool),
    MoveDown(bool),
    MoveLeft(bool),
    MoveRight(bool),
    Confirm,
    Cancel,
    Quit,
}