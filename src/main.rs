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
    Quit,
}

// ============================================
// Raw mode
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

fn start_keyboard_reader(tx: std::sync::mpsc::Sender<UserCommand>, running: Arc<AtomicBool>) {
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

// ============================================
// Статистика
// ============================================

struct TimingStats {
    intervals: VecDeque<u64>,
    conv_times: VecDeque<u64>,
    track_times: VecDeque<u64>,
}

impl TimingStats {
    fn new() -> Self {
        Self {
            intervals: VecDeque::with_capacity(120),
            conv_times: VecDeque::with_capacity(120),
            track_times: VecDeque::with_capacity(120),
        }
    }

    fn add_interval(&mut self, v: u64) {
        if self.intervals.len() >= 120 {
            self.intervals.pop_front();
        }
        self.intervals.push_back(v);
    }

    fn add_times(&mut self, conv: u64, track: u64) {
        if self.conv_times.len() >= 120 {
            self.conv_times.pop_front();
        }
        if self.track_times.len() >= 120 {
            self.track_times.pop_front();
        }
        self.conv_times.push_back(conv);
        self.track_times.push_back(track);
    }

    fn fps(&self) -> f64 {
        if self.intervals.is_empty() {
            return 0.0;
        }
        let avg = self.intervals.iter().sum::<u64>() as f64 / self.intervals.len() as f64;
        if avg > 0.0 {
            1_000_000.0 / avg
        } else {
            0.0
        }
    }

    fn avg_conv_ms(&self) -> f64 {
        if self.conv_times.is_empty() {
            return 0.0;
        }
        self.conv_times.iter().sum::<u64>() as f64 / self.conv_times.len() as f64 / 1000.0
    }

    fn avg_track_ms(&self) -> f64 {
        if self.track_times.is_empty() {
            return 0.0;
        }
        self.track_times.iter().sum::<u64>() as f64 / self.track_times.len() as f64 / 1000.0
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
// Контекст трекера
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
        let tracker = VitTrack::new(model_path).map_err(|e| anyhow!("Failed: {:?}", e))?;
        println!("Model loaded successfully\r");

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
                self.pending_confirm = true;
            }
            UserCommand::Cancel => {
                self.state = AppState::Selecting;
                self.selection = SelectionState::new(self.frame_width, self.frame_height);
                self.current_bbox = None;
                println!("\rReset to selection mode\r");
            }
            UserCommand::Quit => {}
        }
    }

    /// Обрабатывает кадр - принимает ПОЛНЫЙ кадр в BGR
    fn process_frame(&mut self, full_image: &ArrayView3<u8>) -> Option<BBox> {
        match self.state {
            AppState::Selecting => {
                if self.pending_confirm {
                    self.pending_confirm = false;

                    match self.selection.phase {
                        SelectionPhase::MovingToStart => {
                            self.selection.start_x = self.selection.cursor_x;
                            self.selection.start_y = self.selection.cursor_y;
                            self.selection.phase = SelectionPhase::SelectingArea;
                            println!(
                                "\r*** Start point set at ({}, {}) ***\r",
                                self.selection.start_x, self.selection.start_y
                            );
                            println!("\rNow move to the SECOND corner and press Enter\r");
                        }
                        SelectionPhase::SelectingArea => {
                            let bbox = self.selection.get_bbox();
                            println!(
                                "\r*** Initializing tracker with bbox: x={}, y={}, w={}, h={} ***\r",
                                bbox.x, bbox.y, bbox.width, bbox.height
                            );

                            self.tracker.init(full_image, bbox);

                            match self.tracker.update(full_image) {
                                Ok(result) => {
                                    println!("\rInit result: score={:.3}\r", result.score);
                                    if result.success && result.score > 0.25 {
                                        self.current_bbox = Some(BBox::from_array(&result.bbox));
                                        self.current_score = result.score;
                                        self.state = AppState::Tracking;
                                        println!("\r*** TRACKING STARTED! ***\r");
                                        return self.current_bbox;
                                    } else {
                                        println!("\rLow score - please try selecting a different area\r");
                                        self.selection =
                                            SelectionState::new(self.frame_width, self.frame_height);
                                    }
                                }
                                Err(e) => {
                                    println!("\rTracker error: {:?}\r", e);
                                    self.selection =
                                        SelectionState::new(self.frame_width, self.frame_height);
                                }
                            }
                        }
                    }
                }
                None
            }

            AppState::Tracking => {
                self.pending_confirm = false;

                match self.tracker.update(full_image) {
                    Ok(result) => {
                        if result.success && result.score > 0.25 {
                            let bbox = BBox::from_array(&result.bbox);
                            self.current_bbox = Some(bbox);
                            self.current_score = result.score;
                            Some(bbox)
                        } else {
                            println!("\rTrack lost (score={:.2})\r", result.score);
                            self.state = AppState::Lost { frames: 0 };
                            self.current_score = 0.0;
                            None
                        }
                    }
                    Err(_) => {
                        println!("\rTracker error\r");
                        self.state = AppState::Lost { frames: 0 };
                        None
                    }
                }
            }

            AppState::Lost { frames } => {
                self.pending_confirm = false;
                if frames > 60 {
                    println!("\rAuto-reset to selection mode\r");
                    self.state = AppState::Selecting;
                    self.selection = SelectionState::new(self.frame_width, self.frame_height);
                    self.current_bbox = None;
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

fn draw_selection(data: &mut [u8], w: usize, h: usize, sel: &SelectionState) {
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

    let width = 1920;
    let heigth = 1080;
    let caps = gst::ElementFactory::make("capsfilter").build()?;
    caps.set_property(
        "caps",
        &gst::Caps::builder("video/x-raw")
            .field("format", "NV12")
            .field("width", width)
            .field("height", heigth)
            .field("framerate", gst::Fraction::new(60, 1))
            .build(),
    );

    let identity = gst::ElementFactory::make("identity").build()?;
    let convert = gst::ElementFactory::make("videoconvert").build()?;
    let sink = gst::ElementFactory::make("autovideosink")
        .property("sync", false)
        .build()?;

    pipeline.add_many([&src, &caps, &identity, &convert, &sink])?;
    gst::Element::link_many([&src, &caps, &identity, &convert, &sink])?;

    let ctx = Arc::new(Mutex::new(TrackerContext::new(MODEL_PATH, width, heigth)?));
    let stats = Arc::new(Mutex::new(TimingStats::new()));

    let ctx_clone = ctx.clone();
    let stats_clone = stats.clone();
    let cmd_rx = Arc::new(Mutex::new(cmd_rx));

    let frame_num = Arc::new(AtomicU64::new(0));
    let last_time: Arc<Mutex<Option<Instant>>> = Arc::new(Mutex::new(None));

    let pad = identity.static_pad("src").unwrap();

    pad.add_probe(gst::PadProbeType::BUFFER, move |_pad, probe_info| {
        // Замер интервала
        let now = Instant::now();
        {
            let mut last = last_time.lock().unwrap();
            if let Some(prev) = *last {
                stats_clone
                    .lock()
                    .unwrap()
                    .add_interval(now.duration_since(prev).as_micros() as u64);
            }
            *last = Some(now);
        }

        let num = frame_num.fetch_add(1, Ordering::SeqCst);

        // Обработка команд
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

        // Конвертация NV12 -> BGR (полный кадр, но параллельно)
        let t0 = Instant::now();
        let bgr = nv12_full_to_bgr_parallel(data, w, h);
        let conv_time = t0.elapsed().as_micros() as u64;

        // Трекинг
        let t1 = Instant::now();
        let (bbox, state_name, score, selection) = {
            let mut ctx = ctx_clone.lock().unwrap();
            let result = ctx.process_frame(&bgr.view());
            (
                result,
                ctx.state_name().to_string(),
                ctx.current_score,
                ctx.selection.clone(),
            )
        };
        let track_time = t1.elapsed().as_micros() as u64;

        stats_clone.lock().unwrap().add_times(conv_time, track_time);

        // Отрисовка
        draw_background_nv12(data, w, h, 10, 10, 400, 80, 180);
        draw_text_nv12(data, w, h, &state_name, 15, 15, 2, 255);

        let (fps, conv_ms, track_ms) = {
            let s = stats_clone.lock().unwrap();
            (s.fps(), s.avg_conv_ms(), s.avg_track_ms())
        };

        draw_text_nv12(data, w, h, &format!("FPS: {:.0}", fps), 15, 40, 2, 255);
        draw_text_nv12(
            data,
            w,
            h,
            &format!("conv:{:.1}ms trk:{:.1}ms", conv_ms, track_ms),
            15,
            65,
            1,
            200,
        );

        if state_name == "TRACKING" {
            draw_text_nv12(
                data,
                w,
                h,
                &format!("score: {:.0}%", score * 100.0),
                250,
                15,
                2,
                255,
            );
        }

        // Курсор и выделение в режиме выбора
        if state_name.starts_with("SELECT") {
            draw_cursor(data, w, h, selection.cursor_x, selection.cursor_y);
            draw_selection(data, w, h, &selection);
        }

        // BBox
        if let Some(b) = bbox {
            draw_rect_nv12(data, w, h, b.x, b.y, b.width, b.height, 3, 255);
            draw_crosshair_nv12(data, w, h, b.x + b.width / 2, b.y + b.height / 2, 15, 255);
        } else if state_name == "TRACKING" {
            // Показываем последний известный bbox
            if let Some(b) = ctx_clone.lock().unwrap().current_bbox {
                draw_rect_nv12(data, w, h, b.x, b.y, b.width, b.height, 3, 255);
                draw_crosshair_nv12(data, w, h, b.x + b.width / 2, b.y + b.height / 2, 15, 255);
            }
        }

        if num % 120 == 0 && num > 0 {
            println!(
                "\r[{}] FPS: {:.0} | conv: {:.1}ms | track: {:.1}ms\r",
                state_name, fps, conv_ms, track_ms
            );
        }

        gst::PadProbeReturn::Ok
    });

    Ok((pipeline, ctx, stats))
}

fn main() -> Result<()> {
    println!("==========================================\r");
    println!("   VitTrack - Interactive Selection\r");
    println!("==========================================\r\n");

    let device = "/dev/video11";

    if !Path::new(device).exists() {
        return Err(anyhow!("Camera not found: {}", device));
    }

    if !Path::new(MODEL_PATH).exists() {
        return Err(anyhow!("Model not found: {}", MODEL_PATH));
    }

    // Инициализируем rayon
    rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build_global()
        .ok();

    let (tx, rx) = std::sync::mpsc::channel();
    let (pipeline, _ctx, _stats) = create_pipeline(device, rx)?;

    pipeline.set_state(gst::State::Playing)?;

    let running = Arc::new(AtomicBool::new(true));
    start_keyboard_reader(tx, running.clone());

    let bus = pipeline.bus().unwrap();

    while running.load(Ordering::SeqCst) {
        if let Some(msg) = bus.timed_pop(gst::ClockTime::from_mseconds(100)) {
            if let gst::MessageView::Error(err) = msg.view() {
                eprintln!("\rError: {}\r", err.error());
                break;
            }
        }
    }

    pipeline.set_state(gst::State::Null)?;
    println!("\r\nDone\r");

    Ok(())
}