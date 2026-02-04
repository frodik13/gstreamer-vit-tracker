use anyhow::{anyhow, Result};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::io::{self, Read};

use crate::user_commands::UserCommand;

pub struct RawModeGuard {
    pub(crate) original: libc::termios,
}

impl RawModeGuard {
    pub fn new() -> Result<Self> {
        unsafe {
            let mut original: libc::termios = std::mem::zeroed();
            if libc::tcgetattr(0, &mut original) != 0 {
                return Err(anyhow!("tcgetattr failed"));
            }
            let mut raw = original;
            raw.c_lflag &= !(libc::ICANON | libc::ECHO);
            raw.c_cc[libc::VMIN] = 1;
            raw.c_cc[libc::VTIME] = 0;
            if libc::tcsetattr(0, libc::TCSANOW, &raw) != 0 {
                return Err(anyhow!("tcsetattr failed"));
            }
            Ok(Self { original })
        }
    }
}

impl Drop for RawModeGuard {
    fn drop(&mut self) {
        unsafe {
            libc::tcsetattr(0, libc::TCSANOW, &self.original);
        }
    }
}

pub fn start_keyboard_reader(tx: std::sync::mpsc::Sender<UserCommand>, running: Arc<AtomicBool>) {
    std::thread::spawn(move || {
        let _guard = RawModeGuard::new().ok();

        println!("\r");
        println!("╔═══════════════════════════════════════════╗\r");
        println!("║            KEYBOARD CONTROLS              ║\r");
        println!("╠═══════════════════════════════════════════╣\r");
        println!("║  W/A/S/D or I/J/K/L  - Move cursor        ║\r");
        println!("║  Shift + above       - Fast move          ║\r");
        println!("║  Enter or Space      - Confirm point      ║\r");
        println!("║  R or Escape         - Reset              ║\r");
        println!("║  Q                   - Quit               ║\r");
        println!("╚═══════════════════════════════════════════╝\r");
        println!("\r");
        println!("Step 1: Move to FIRST corner, press Enter\r");
        println!("Step 2: Move to SECOND corner, press Enter\r");
        println!("\r");

        let stdin = io::stdin();

        for byte in stdin.lock().bytes().flatten() {
            if !running.load(Ordering::SeqCst) {
                break;
            }

            let cmd = match byte {
                // Enter, Space - confirm
                10 | 13 | 32 => Some(UserCommand::Confirm),

                // W, w, I, i - up
                87 | 119 | 73 | 105 => Some(UserCommand::MoveUp(false)),
                // S, s, K, k - down
                83 | 115 | 75 | 107 => Some(UserCommand::MoveDown(false)),
                // A, a, J, j - left
                65 | 97 | 74 | 106 => Some(UserCommand::MoveLeft(false)),
                // D, d, L, l - right
                68 | 100 | 76 | 108 => Some(UserCommand::MoveRight(false)),

                // Fast movement: T, G, F, H (or shift variants)
                84 | 116 => Some(UserCommand::MoveUp(true)),    // T, t
                71 | 103 => Some(UserCommand::MoveDown(true)),  // G, g
                70 | 102 => Some(UserCommand::MoveLeft(true)),  // F, f
                72 | 104 => Some(UserCommand::MoveRight(true)), // H, h

                // Arrow keys (escape sequences) - we read byte by byte
                // Up=65, Down=66, Right=67, Left=68 after ESC [
                // But since we read byte by byte, arrows come as separate bytes

                // R, r, Escape - reset
                82 | 114 | 27 => Some(UserCommand::Cancel),

                // Q, q - quit
                81 | 113 => {
                    running.store(false, Ordering::SeqCst);
                    Some(UserCommand::Quit)
                }

                // Ignore escape sequence parts
                91 => None, // [

                _ => None,
            };

            if let Some(c) = cmd {
                let _ = tx.send(c);
            }
        }
    });
}