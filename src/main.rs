mod nv12_convert;
mod rga;

use anyhow::{anyhow, Result};
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_video as gst_video;
use ndarray::{Array3, ArrayView3};
use nv12_convert::*;
use std::collections::VecDeque;
use std::io::{self, Read};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use vit_tracker::{BBox, VitTrack};

const MODEL_PATH: &str = "/home/radxa/repos/rust/vit_tracker/models/object_tracking_vittrack_2023sep.rknn";

// ============================================
// Состояние выделения
// ============================================

#[derive(Debug, Clone, Copy, PartialEq)]
enum SelectionPhase {
    MovingToStart,
    SelectingArea,
}

#[derive(Debug, Clone)]
struct SelectionState {
    cursor_x: i32,
    cursor_y: i32,
    start_x: i32,
    start_y: i32,
    phase: SelectionPhase,
    step: i32,
    fast_step: i32,
}

impl SelectionState {
    fn new(width: i32, height: i32) -> Self {
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

    fn move_cursor(&mut self, dx: i32, dy: i32, fast: bool, width: i32, height: i32) {
        let step = if fast { self.fast_step } else { self.step };
        self.cursor_x = (self.cursor_x + dx * step).clamp(0, width - 1);
        self.cursor_y = (self.cursor_y + dy * step).clamp(0, height - 1);
    }

    fn get_bbox(&self) -> BBox {
        let x = self.start_x.min(self.cursor_x);
        let y = self.start_y.min(self.cursor_y);
        let w = (self.start_x - self.cursor_x).abs().max(20);
        let h = (self.start_y - self.cursor_y).abs().max(20);
        BBox::new(x, y, w, h)
    }
}

// ============================================
// Команды
// ============================================

#[derive(Debug, Clone)]
enum UserCommand {
    MoveUp(bool),
    MoveDown(bool),
    MoveLeft(bool),
    MoveRight(bool),
    Confirm,
    Cancel,
    Reset,
    Quit,
}

// ============================================
// Raw mode терминала
// ============================================

struct RawModeGuard {
    original: libc::termios,
}

impl RawModeGuard {
    fn new() -> Result<Self> {
        unsafe {
            let mut original: libc::termios = std::mem::zeroed();
            if libc::tcgetattr(0, &mut original) != 0 {
                return Err(anyhow!("tcgetattr failed"));
            }

            let mut raw = original;
            raw.c_lflag &= !(libc::ICANON | libc::ECHO);
            raw.c_cc[libc::VMIN] = 1;  // Ждём хотя бы 1 байт
            raw.c_cc[libc::VTIME] = 0; // Без таймаута

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

fn start_keyboard_reader(tx: std::sync::mpsc::Sender<UserCommand>, running: Arc<AtomicBool>) {
    std::thread::spawn(move || {
        let _guard = match RawModeGuard::new() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("Failed to set raw mode: {}", e);
                return;
            }
        };

        println!("\r");
        println!("╔══════════════════════════════════════╗\r");
        println!("║       KEYBOARD CONTROLS              ║\r");
        println!("╠══════════════════════════════════════╣\r");
        println!("║ Arrows      - Move cursor            ║\r");
        println!("║ Shift+Arrow - Fast move              ║\r");
        println!("║ Enter       - Set point / Confirm    ║\r");
        println!("║ Escape      - Cancel                 ║\r");
        println!("║ R           - Reset tracker          ║\r");
        println!("║ Q           - Quit                   ║\r");
        println!("╚══════════════════════════════════════╝\r");
        println!("\r");

        let stdin = io::stdin();
        
        for byte_result in stdin.lock().bytes() {
            if !running.load(Ordering::SeqCst) {
                break;
            }

            let byte = match byte_result {
                Ok(b) => b,
                Err(_) => continue,
            };

            // Отладка: показываем что получили
            println!("\rKey byte: {} (0x{:02x})\r", byte, byte);

            let cmd = match byte {
                // Enter (разные варианты)
                10 | 13 => {
                    println!("\r>>> ENTER pressed! <<<\r");
                    Some(UserCommand::Confirm)
                }
                
                // Escape
                27 => {
                    // Читаем следующие байты для escape sequences
                    let mut seq = vec![27u8];
                    
                    // Небольшая пауза чтобы прочитать остаток последовательности
                    std::thread::sleep(std::time::Duration::from_millis(10));
                    
                    // Пробуем прочитать ещё
                    // Это упрощённый вариант - для стрелок
                    None // Пока игнорируем escape sequences
                }
                
                // Стрелки через ANSI (когда терминал отправляет отдельные байты)
                // Обычно это Esc [ A/B/C/D
                65 => Some(UserCommand::MoveUp(false)),    // 'A' после Esc[
                66 => Some(UserCommand::MoveDown(false)),  // 'B'
                67 => Some(UserCommand::MoveRight(false)), // 'C'
                68 => Some(UserCommand::MoveLeft(false)),  // 'D'
                
                // WASD как альтернатива стрелкам
                119 | 87 => Some(UserCommand::MoveUp(false)),    // w/W
                115 | 83 => Some(UserCommand::MoveDown(false)),  // s/S
                97 | 65 => Some(UserCommand::MoveLeft(false)),   // a/A - конфликт с Up!
                100 | 68 => Some(UserCommand::MoveRight(false)), // d/D - конфликт с Left!
                
                // IJKL как альтернатива (без конфликтов)
                105 | 73 => Some(UserCommand::MoveUp(false)),    // i/I
                107 | 75 => Some(UserCommand::MoveDown(false)),  // k/K
                106 | 74 => Some(UserCommand::MoveLeft(false)),  // j/J
                108 | 76 => Some(UserCommand::MoveRight(false)), // l/L
                
                // Быстрое перемещение: TFGH
                116 | 84 => Some(UserCommand::MoveUp(true)),    // t/T
                103 | 71 => Some(UserCommand::MoveDown(true)),  // g/G
                102 | 70 => Some(UserCommand::MoveLeft(true)),  // f/F
                104 | 72 => Some(UserCommand::MoveRight(true)), // h/H
                
                // R - reset
                114 | 82 => {
                    println!("\r>>> RESET pressed! <<<\r");
                    Some(UserCommand::Reset)
                }
                
                // Q - quit
                113 | 81 => {
                    println!("\r>>> QUIT pressed! <<<\r");
                    running.store(false, Ordering::SeqCst);
                    Some(UserCommand::Quit)
                }
                
                // Space - тоже как confirm
                32 => {
                    println!("\r>>> SPACE pressed (confirm)! <<<\r");
                    Some(UserCommand::Confirm)
                }
                
                // [ - часть escape sequence, игнорируем
                91 => None,
                
                _ => None,
            };

            if let Some(cmd) = cmd {
                if tx.send(cmd).is_err() {
                    break;
                }
            }
        }
    });
}

// ============================================
// Статистика
// ============================================

struct TimingStats {
    frame_intervals_us: VecDeque<u64>,
    max_samples: usize,
}

impl TimingStats {
    fn new(max_samples: usize) -> Self {
        Self {
            frame_intervals_us: VecDeque::with_capacity(max_samples),
            max_samples,
        }
    }

    fn add_interval(&mut self, interval: u64) {
        if self.frame_intervals_us.len() >= self.max_samples {
            self.frame_intervals_us.pop_front();
        }
        self.frame_intervals_us.push_back(interval);
    }

    fn fps(&self) -> f64 {
        if self.frame_intervals_us.is_empty() {
            return 0.0;
        }
        let avg = self.frame_intervals_us.iter().sum::<u64>() as f64
            / self.frame_intervals_us.len() as f64;
        if avg > 0.0 { 1_000_000.0 / avg } else { 0.0 }
    }
}

// ============================================
// Состояние приложения
// ============================================

#[derive(Debug, Clone, Copy, PartialEq)]
enum AppState {
    Selecting,
    Tracking,
    Lost { frames: u64 },
}

// ============================================
// Контекст
// ============================================

struct TrackerContext {
    tracker: VitTrack,
    state: AppState,
    selection: SelectionState,
    current_bbox: Option<BBox>,
    current_score: f32,
    frame_width: i32,
    frame_height: i32,
    pending_confirm: bool,
}

impl TrackerContext {
    fn new(model_path: &str, width: i32, height: i32) -> Result<Self> {
        println!("Loading model: {}\r", model_path);
        let tracker = VitTrack::new(model_path)
            .map_err(|e| anyhow!("Failed: {:?}", e))?;
        println!("Model loaded\r");

        Ok(Self {
            tracker,
            state: AppState::Selecting,
            selection: SelectionState::new(width, height),
            current_bbox: None,
            current_score: 0.0,
            frame_width: width,
            frame_height: height,
            pending_confirm: false,
        })
    }

    fn handle_command(&mut self, cmd: UserCommand) {
        match cmd {
            UserCommand::MoveUp(fast) => {
                self.selection.move_cursor(0, -1, fast, self.frame_width, self.frame_height);
            }
            UserCommand::MoveDown(fast) => {
                self.selection.move_cursor(0, 1, fast, self.frame_width, self.frame_height);
            }
            UserCommand::MoveLeft(fast) => {
                self.selection.move_cursor(-1, 0, fast, self.frame_width, self.frame_height);
            }
            UserCommand::MoveRight(fast) => {
                self.selection.move_cursor(1, 0, fast, self.frame_width, self.frame_height);
            }
            UserCommand::Confirm => {
                println!("\rConfirm command received!\r");
                self.pending_confirm = true;
            }
            UserCommand::Cancel | UserCommand::Reset => {
                self.state = AppState::Selecting;
                self.selection = SelectionState::new(self.frame_width, self.frame_height);
                self.current_bbox = None;
                println!("\rReset to selection mode\r");
            }
            UserCommand::Quit => {}
        }
    }

    fn process_frame(&mut self, image: &ArrayView3<u8>) -> Option<BBox> {
        match self.state {
            AppState::Selecting => {
                if self.pending_confirm {
                    self.pending_confirm = false;
                    println!("\rProcessing confirm in phase: {:?}\r", self.selection.phase);

                    match self.selection.phase {
                        SelectionPhase::MovingToStart => {
                            self.selection.start_x = self.selection.cursor_x;
                            self.selection.start_y = self.selection.cursor_y;
                            self.selection.phase = SelectionPhase::SelectingArea;
                            println!("\r*** Start point: ({}, {}) ***\r",
                                     self.selection.start_x, self.selection.start_y);
                            println!("\r*** Now move to second corner and press ENTER/SPACE ***\r");
                        }
                        SelectionPhase::SelectingArea => {
                            let bbox = self.selection.get_bbox();
                            println!("\r*** Initializing tracker: {:?} ***\r", bbox);

                            self.tracker.init(image, bbox);

                            match self.tracker.update(image) {
                                Ok(result) => {
                                    println!("\rInit score: {:.3}\r", result.score);
                                    if result.success && result.score > 0.25 {
                                        self.current_bbox = Some(BBox::from_array(&result.bbox));
                                        self.current_score = result.score;
                                        self.state = AppState::Tracking;
                                        println!("\r*** TRACKING STARTED! ***\r");
                                        return self.current_bbox;
                                    } else {
                                        println!("\rLow score - try different area\r");
                                        self.selection = SelectionState::new(self.frame_width, self.frame_height);
                                    }
                                }
                                Err(e) => {
                                    println!("\rError: {:?}\r", e);
                                    self.selection = SelectionState::new(self.frame_width, self.frame_height);
                                }
                            }
                        }
                    }
                }
                None
            }

            AppState::Tracking => {
                self.pending_confirm = false;

                match self.tracker.update(image) {
                    Ok(result) => {
                        if result.success && result.score > 0.25 {
                            let bbox = BBox::from_array(&result.bbox);
                            self.current_bbox = Some(bbox);
                            self.current_score = result.score;
                            Some(bbox)
                        } else {
                            println!("\rLost (score={:.2})\r", result.score);
                            self.state = AppState::Lost { frames: 0 };
                            None
                        }
                    }
                    Err(_) => {
                        self.state = AppState::Lost { frames: 0 };
                        None
                    }
                }
            }

            AppState::Lost { frames } => {
                self.pending_confirm = false;
                if frames > 120 {
                    self.state = AppState::Selecting;
                    self.selection = SelectionState::new(self.frame_width, self.frame_height);
                } else {
                    self.state = AppState::Lost { frames: frames + 1 };
                }
                None
            }
        }
    }

    fn state_name(&self) -> &'static str {
        match self.state {
            AppState::Selecting => match self.selection.phase {
                SelectionPhase::MovingToStart => "SELECT START",
                SelectionPhase::SelectingArea => "SELECT END",
            },
            AppState::Tracking => "TRACKING",
            AppState::Lost { .. } => "LOST",
        }
    }
}

// ============================================
// Отрисовка
// ============================================

fn draw_cursor(data: &mut [u8], w: usize, h: usize, x: i32, y: i32) {
    let x = x.clamp(0, w as i32 - 1) as usize;
    let y = y.clamp(0, h as i32 - 1) as usize;
    let plane = &mut data[..w * h];

    for px in x.saturating_sub(25)..=(x + 25).min(w - 1) {
        if !(x.saturating_sub(5)..=x + 5).contains(&px) {
            plane[y * w + px] = 255;
        }
    }

    for py in y.saturating_sub(25)..=(y + 25).min(h - 1) {
        if !(y.saturating_sub(5)..=y + 5).contains(&py) {
            plane[py * w + x] = 255;
        }
    }
}

fn draw_selection(data: &mut [u8], w: usize, h: usize, sel: &SelectionState) {
    if sel.phase != SelectionPhase::SelectingArea {
        return;
    }

    let x1 = sel.start_x.min(sel.cursor_x).max(0) as usize;
    let y1 = sel.start_y.min(sel.cursor_y).max(0) as usize;
    let x2 = (sel.start_x.max(sel.cursor_x) as usize).min(w - 1);
    let y2 = (sel.start_y.max(sel.cursor_y) as usize).min(h - 1);

    let plane = &mut data[..w * h];

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

// ============================================
// Pipeline
// ============================================

fn create_pipeline(
    device: &str,
    cmd_rx: std::sync::mpsc::Receiver<UserCommand>,
) -> Result<(gst::Pipeline, Arc<Mutex<TrackerContext>>, Arc<Mutex<TimingStats>>)> {
    gst::init()?;

    let pipeline = gst::Pipeline::new();

    let src = gst::ElementFactory::make("v4l2src")
        .property("device", device)
        .property("do-timestamp", true)
        .build()?;

    let caps = gst::ElementFactory::make("capsfilter").build()?;
    caps.set_property("caps", &gst::Caps::builder("video/x-raw")
        .field("format", "NV12")
        .field("width", 1920i32)
        .field("height", 1080i32)
        .field("framerate", gst::Fraction::new(60, 1))
        .build());

    let identity = gst::ElementFactory::make("identity").build()?;
    let convert = gst::ElementFactory::make("videoconvert").build()?;
    let sink = gst::ElementFactory::make("autovideosink")
        .property("sync", false)
        .build()?;

    pipeline.add_many([&src, &caps, &identity, &convert, &sink])?;
    gst::Element::link_many([&src, &caps, &identity, &convert, &sink])?;

    let ctx = Arc::new(Mutex::new(TrackerContext::new(MODEL_PATH, 1920, 1080)?));
    let stats = Arc::new(Mutex::new(TimingStats::new(120)));

    let ctx_clone = ctx.clone();
    let stats_clone = stats.clone();
    let cmd_rx = Arc::new(Mutex::new(cmd_rx));

    let frame_num = Arc::new(AtomicU64::new(0));
    let last_time: Arc<Mutex<Option<Instant>>> = Arc::new(Mutex::new(None));

    let pad = identity.static_pad("src").unwrap();

    pad.add_probe(gst::PadProbeType::BUFFER, move |_pad, probe_info| {
        let now = Instant::now();
        {
            let mut last = last_time.lock().unwrap();
            if let Some(prev) = *last {
                stats_clone.lock().unwrap().add_interval(
                    now.duration_since(prev).as_micros() as u64
                );
            }
            *last = Some(now);
        }

        let _num = frame_num.fetch_add(1, Ordering::SeqCst);

        // Команды
        if let Ok(rx) = cmd_rx.lock() {
            while let Ok(cmd) = rx.try_recv() {
                ctx_clone.lock().unwrap().handle_command(cmd);
            }
        }

        let buffer = match &mut probe_info.data {
            Some(gst::PadProbeData::Buffer(b)) => b,
            _ => return gst::PadProbeReturn::Ok,
        };

        let buffer = buffer.make_mut();
        let Ok(mut map) = buffer.map_writable() else {
            return gst::PadProbeReturn::Ok;
        };

        let data = map.as_mut_slice();
        let (w, h) = (1920usize, 1080usize);

        let bgr = nv12_full_to_bgr(data, w, h);

        let (bbox, state_name, score, selection) = {
            let mut ctx = ctx_clone.lock().unwrap();
            let result = ctx.process_frame(&bgr.view());
            (result, ctx.state_name().to_string(), ctx.current_score, ctx.selection.clone())
        };

        // Отрисовка
        draw_background_nv12(data, w, h, 10, 10, 350, 80, 180);
        draw_text_nv12(data, w, h, &state_name, 15, 15, 2, 255);

        let fps = stats_clone.lock().unwrap().fps();
        draw_text_nv12(data, w, h, &format!("FPS: {:.0}", fps), 15, 40, 2, 255);
        draw_text_nv12(data, w, h, &format!("pos: {}.{}", selection.cursor_x, selection.cursor_y), 15, 65, 1, 200);

        if state_name == "TRACKING" {
            draw_text_nv12(data, w, h, &format!("score: {:.0}%", score * 100.0), 200, 15, 2, 255);
        }

        if state_name.starts_with("SELECT") {
            draw_cursor(data, w, h, selection.cursor_x, selection.cursor_y);
            draw_selection(data, w, h, &selection);
        }

        if let Some(b) = bbox {
            draw_rect_nv12(data, w, h, b.x, b.y, b.width, b.height, 3, 255);
            draw_crosshair_nv12(data, w, h, b.x + b.width / 2, b.y + b.height / 2, 15, 255);
        }

        gst::PadProbeReturn::Ok
    });

    Ok((pipeline, ctx, stats))
}

fn main() -> Result<()> {
    println!("==========================================");
    println!("   VitTrack - Interactive Selection");
    println!("==========================================\n");

    let device = "/dev/video11";

    if !Path::new(device).exists() {
        return Err(anyhow!("Camera not found: {}", device));
    }

    if !Path::new(MODEL_PATH).exists() {
        return Err(anyhow!("Model not found: {}", MODEL_PATH));
    }

    let (tx, rx) = std::sync::mpsc::channel();
    let (pipeline, _ctx, _stats) = create_pipeline(device, rx)?;

    pipeline.set_state(gst::State::Playing)?;

    let running = Arc::new(AtomicBool::new(true));
    start_keyboard_reader(tx, running.clone());

    let bus = pipeline.bus().unwrap();

    while running.load(Ordering::SeqCst) {
        if let Some(msg) = bus.timed_pop(gst::ClockTime::from_mseconds(100)) {
            use gst::MessageView;
            if let MessageView::Error(err) = msg.view() {
                eprintln!("\rError: {}\r", err.error());
                break;
            }
        }
    }

    pipeline.set_state(gst::State::Null)?;
    println!("\r\nDone\r");

    Ok(())
}