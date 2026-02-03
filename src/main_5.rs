use anyhow::{anyhow, Result};
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_video as gst_video;
use ndarray::{Array3, ArrayView3};
use std::collections::VecDeque;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

// ============================================
// Импорт вашего трекера - раскомментируйте
// ============================================
use vit_tracker::{VitTrack, tracker::VitTrackConfig, BBox, TrackingResult};

// ============================================
// Детальные замеры времени
// ============================================

#[derive(Default, Clone)]
struct TimingStats {
    copy_times_us: VecDeque<u64>,
    track_times_us: VecDeque<u64>,
    draw_times_us: VecDeque<u64>,
    total_times_us: VecDeque<u64>,
    max_samples: usize,
}

impl TimingStats {
    fn new(max_samples: usize) -> Self {
        Self {
            copy_times_us: VecDeque::with_capacity(max_samples),
            track_times_us: VecDeque::with_capacity(max_samples),
            draw_times_us: VecDeque::with_capacity(max_samples),
            total_times_us: VecDeque::with_capacity(max_samples),
            max_samples,
        }
    }

    fn add(&mut self, copy_us: u64, track_us: u64, draw_us: u64, total_us: u64) {
        Self::push(&mut self.copy_times_us, copy_us, self.max_samples);
        Self::push(&mut self.track_times_us, track_us, self.max_samples);
        Self::push(&mut self.draw_times_us, draw_us, self.max_samples);
        Self::push(&mut self.total_times_us, total_us, self.max_samples);
    }

    fn push(queue: &mut VecDeque<u64>, value: u64, max: usize) {
        if queue.len() >= max {
            queue.pop_front();
        }
        queue.push_back(value);
    }

    fn avg(queue: &VecDeque<u64>) -> f64 {
        if queue.is_empty() { 0.0 } else {
            queue.iter().sum::<u64>() as f64 / queue.len() as f64 / 1000.0
        }
    }

    fn report(&self) -> String {
        format!(
            "copy: {:.2}ms | track: {:.2}ms | draw: {:.2}ms | total: {:.2}ms",
            Self::avg(&self.copy_times_us),
            Self::avg(&self.track_times_us),
            Self::avg(&self.draw_times_us),
            Self::avg(&self.total_times_us),
        )
    }
}

// ============================================
// Состояние трекера (исправленная версия)
// ============================================

#[derive(Debug, Clone, Copy, PartialEq)]
enum AppState {
    WaitingForInit { frames_waited: u64 },
    Tracking,
    Lost { frames_lost: u64 },
}

impl AppState {
    fn name(&self) -> &'static str {
        match self {
            AppState::WaitingForInit { .. } => "WAITING",
            AppState::Tracking => "TRACKING",
            AppState::Lost { .. } => "LOST",
        }
    }
}

#[derive(Debug, Clone)]
pub enum InitMode {
    Fixed(BBox),
    Center { width: i32, height: i32 },
    DelayedCenter { delay_frames: u64, width: i32, height: i32 },
}

#[derive(Debug, Clone)]
pub struct InitialBBoxConfig {
    pub mode: InitMode,
}

impl Default for InitialBBoxConfig {
    fn default() -> Self {
        Self {
            mode: InitMode::DelayedCenter { delay_frames: 60, width: 150, height: 150 },
        }
    }
}

struct TrackerContext {
    tracker: VitTrack,
    state: AppState,
    init_config: InitialBBoxConfig,
    current_bbox: Option<BBox>,
    current_score: f32,
    timing: TimingStats,
    total_tracked_frames: u64,
    lost_count: u64,
}

impl TrackerContext {
    fn new(config: VitTrackConfig, init_config: InitialBBoxConfig) -> Result<Self> {
        let model_path = Path::new("/home")
            .join("radxa")
            .join("repos")
            .join("rust")
            .join("vit_tracker")
            .join("models")
            .join("object_tracking_vittrack_2023sep.rknn");
        let tracker = VitTrack::new(model_path.to_str().unwrap())
            .map_err(|_| anyhow!("Не удалось создать трекер"))?;
        Ok(Self {
            tracker,
            state: AppState::WaitingForInit { frames_waited: 0 },
            init_config,
            current_bbox: None,
            current_score: 0.0,
            timing: TimingStats::new(100),
            total_tracked_frames: 0,
            lost_count: 0,
        })
    }

    /// Обработка кадра БЕЗ копирования - используем ArrayView3
    fn process_frame(
        &mut self,
        image: &ArrayView3<u8>,
        frame_num: u64,
    ) -> (Option<BBox>, u64, u64) {
        let (height, width) = (image.shape()[0], image.shape()[1]);
        
        let mut track_time_us = 0u64;
        let mut copy_time_us = 0u64;

        // Читаем текущее состояние
        let current_state = self.state;

        match current_state {
            AppState::WaitingForInit { frames_waited } => {
                let new_frames_waited = frames_waited + 1;

                let init_bbox = match &self.init_config.mode {
                    InitMode::Fixed(bbox) => Some(*bbox),
                    InitMode::Center { width: w, height: h } => Some(BBox::new(
                        (width as i32 - w) / 2,
                        (height as i32 - h) / 2,
                        *w, *h,
                    )),
                    InitMode::DelayedCenter { delay_frames, width: w, height: h } => {
                        if new_frames_waited >= *delay_frames {
                            Some(BBox::new(
                                (width as i32 - w) / 2,
                                (height as i32 - h) / 2,
                                *w, *h,
                            ))
                        } else {
                            None
                        }
                    }
                };

                if let Some(bbox) = init_bbox {
                    let start = Instant::now();
                    self.tracker.init(image, bbox);
                    track_time_us = start.elapsed().as_micros() as u64;
                    
                    self.current_bbox = Some(bbox);
                    self.state = AppState::Tracking;
                    println!("Frame {}: Initialized at {:?}", frame_num, bbox);
                    return (Some(bbox), track_time_us, copy_time_us);
                } else {
                    self.state = AppState::WaitingForInit { frames_waited: new_frames_waited };
                }

                (None, track_time_us, copy_time_us)
            }

            AppState::Tracking => {
                let start = Instant::now();
                let result = self.tracker.update(image);
                track_time_us = start.elapsed().as_micros() as u64;

                match result {
                    Ok(res) if res.success && res.score > 0.3 => {
                        let bbox = BBox::from_array(&res.bbox);
                        self.current_bbox = Some(bbox);
                        self.current_score = res.score;
                        self.total_tracked_frames += 1;
                        (Some(bbox), track_time_us, copy_time_us)
                    }
                    _ => {
                        self.state = AppState::Lost { frames_lost: 0 };
                        self.lost_count += 1;
                        self.current_score = 0.0;
                        println!("Frame {}: Track lost", frame_num);
                        (None, track_time_us, copy_time_us)
                    }
                }
            }

            AppState::Lost { frames_lost } => {
                let new_frames_lost = frames_lost + 1;

                if new_frames_lost > 60 {
                    self.state = AppState::WaitingForInit { frames_waited: 0 };
                    println!("Frame {}: Resetting after {} lost frames", frame_num, new_frames_lost);
                } else {
                    self.state = AppState::Lost { frames_lost: new_frames_lost };
                }

                (None, track_time_us, copy_time_us)
            }
        }
    }

    fn state_name(&self) -> &'static str {
        self.state.name()
    }
}

// ============================================
// FPS Analyzer
// ============================================

struct FpsAnalyzer {
    intervals_us: VecDeque<u64>,
    last_time: Option<Instant>,
    total_frames: u64,
    start_time: Instant,
}

impl FpsAnalyzer {
    fn new(max_samples: usize) -> Self {
        Self {
            intervals_us: VecDeque::with_capacity(max_samples),
            last_time: None,
            total_frames: 0,
            start_time: Instant::now(),
        }
    }

    fn add_frame(&mut self) {
        let now = Instant::now();
        if let Some(last) = self.last_time {
            let interval = now.duration_since(last).as_micros() as u64;
            if self.intervals_us.len() >= self.intervals_us.capacity() {
                self.intervals_us.pop_front();
            }
            self.intervals_us.push_back(interval);
        }
        self.last_time = Some(now);
        self.total_frames += 1;
    }

    fn current_fps(&self) -> f64 {
        if self.intervals_us.is_empty() { return 0.0; }
        let avg = self.intervals_us.iter().sum::<u64>() as f64 / self.intervals_us.len() as f64;
        if avg > 0.0 { 1_000_000.0 / avg } else { 0.0 }
    }

    fn avg_fps(&self) -> f64 {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        if elapsed > 0.0 { self.total_frames as f64 / elapsed } else { 0.0 }
    }
}

// ============================================
// Рисование (оптимизированное)
// ============================================

#[inline(always)]
fn set_pixel_unchecked(buffer: &mut [u8], idx: usize, color: (u8, u8, u8)) {
    buffer[idx] = color.0;
    buffer[idx + 1] = color.1;
    buffer[idx + 2] = color.2;
}

fn draw_rect(buffer: &mut [u8], width: usize, height: usize, bbox: &BBox, color: (u8, u8, u8), thickness: usize) {
    let x1 = bbox.x.max(0) as usize;
    let y1 = bbox.y.max(0) as usize;
    let x2 = ((bbox.x + bbox.width) as usize).min(width.saturating_sub(1));
    let y2 = ((bbox.y + bbox.height) as usize).min(height.saturating_sub(1));

    let stride = width * 3;

    // Горизонтальные линии
    for t in 0..thickness.min(y2 - y1) {
        // Верхняя
        let row_start = (y1 + t) * stride;
        for x in x1..=x2 {
            let idx = row_start + x * 3;
            if idx + 2 < buffer.len() {
                set_pixel_unchecked(buffer, idx, color);
            }
        }
        // Нижняя
        let row_start = (y2 - t) * stride;
        for x in x1..=x2 {
            let idx = row_start + x * 3;
            if idx + 2 < buffer.len() {
                set_pixel_unchecked(buffer, idx, color);
            }
        }
    }

    // Вертикальные линии
    for y in y1..=y2 {
        let row_start = y * stride;
        for t in 0..thickness.min(x2 - x1) {
            // Левая
            let idx = row_start + (x1 + t) * 3;
            if idx + 2 < buffer.len() {
                set_pixel_unchecked(buffer, idx, color);
            }
            // Правая
            let idx = row_start + (x2 - t) * 3;
            if idx + 2 < buffer.len() {
                set_pixel_unchecked(buffer, idx, color);
            }
        }
    }
}

fn draw_crosshair(buffer: &mut [u8], width: usize, height: usize, cx: i32, cy: i32, size: i32, color: (u8, u8, u8)) {
    let cx = cx.max(0) as usize;
    let cy = cy.max(0) as usize;
    let size = size as usize;
    let stride = width * 3;

    // Горизонтальная
    if cy < height {
        let row_start = cy * stride;
        for x in cx.saturating_sub(size)..=(cx + size).min(width - 1) {
            let idx = row_start + x * 3;
            if idx + 2 < buffer.len() {
                set_pixel_unchecked(buffer, idx, color);
            }
        }
    }

    // Вертикальная
    if cx < width {
        for y in cy.saturating_sub(size)..=(cy + size).min(height - 1) {
            let idx = y * stride + cx * 3;
            if idx + 2 < buffer.len() {
                set_pixel_unchecked(buffer, idx, color);
            }
        }
    }
}

fn draw_text(buffer: &mut [u8], width: usize, height: usize, text: &str, x: usize, y: usize, color: (u8, u8, u8), scale: usize) {
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
        ("X", [0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b01010, 0b10001]),
        ("m", [0b00000, 0b00000, 0b11010, 0b10101, 0b10101, 0b10001, 0b10001]),
        ("s", [0b00000, 0b00000, 0b01110, 0b10000, 0b01110, 0b00001, 0b11110]),
        ("c", [0b00000, 0b00000, 0b01110, 0b10000, 0b10000, 0b10001, 0b01110]),
        ("o", [0b00000, 0b00000, 0b01110, 0b10001, 0b10001, 0b10001, 0b01110]),
        ("p", [0b00000, 0b00000, 0b11110, 0b10001, 0b11110, 0b10000, 0b10000]),
        ("y", [0b00000, 0b00000, 0b10001, 0b10001, 0b01111, 0b00001, 0b01110]),
        ("r", [0b00000, 0b00000, 0b10110, 0b11001, 0b10000, 0b10000, 0b10000]),
        ("a", [0b00000, 0b00000, 0b01110, 0b00001, 0b01111, 0b10001, 0b01111]),
        ("w", [0b00000, 0b00000, 0b10001, 0b10001, 0b10101, 0b10101, 0b01010]),
        ("k", [0b10000, 0b10000, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010]),
        ("t", [0b01000, 0b01000, 0b11100, 0b01000, 0b01000, 0b01001, 0b00110]),
        ("d", [0b00001, 0b00001, 0b01111, 0b10001, 0b10001, 0b10001, 0b01111]),
        ("l", [0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110]),
        ("g", [0b00000, 0b00000, 0b01111, 0b10001, 0b01111, 0b00001, 0b01110]),
        ("v", [0b00000, 0b00000, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100]),
        ("i", [0b00100, 0b00000, 0b01100, 0b00100, 0b00100, 0b00100, 0b01110]),
        ("n", [0b00000, 0b00000, 0b10110, 0b11001, 0b10001, 0b10001, 0b10001]),
        ("[", [0b01110, 0b01000, 0b01000, 0b01000, 0b01000, 0b01000, 0b01110]),
        ("]", [0b01110, 0b00010, 0b00010, 0b00010, 0b00010, 0b00010, 0b01110]),
        ("%", [0b11001, 0b11010, 0b00100, 0b00100, 0b01000, 0b01011, 0b10011]),
        ("|", [0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100]),
        ("x", [0b00000, 0b00000, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001]),
    ];

    let stride = width * 3;
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
                                    let idx = py * stride + px * 3;
                                    if idx + 2 < buffer.len() {
                                        set_pixel_unchecked(buffer, idx, color);
                                    }
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

fn draw_background(buffer: &mut [u8], width: usize, height: usize, x: usize, y: usize, w: usize, h: usize, opacity: u8) {
    let stride = width * 3;
    let factor = (255 - opacity) as u16;
    
    for py in y..(y + h).min(height) {
        let row_start = py * stride;
        for px in x..(x + w).min(width) {
            let idx = row_start + px * 3;
            if idx + 2 < buffer.len() {
                buffer[idx] = ((buffer[idx] as u16 * factor) / 255) as u8;
                buffer[idx + 1] = ((buffer[idx + 1] as u16 * factor) / 255) as u8;
                buffer[idx + 2] = ((buffer[idx + 2] as u16 * factor) / 255) as u8;
            }
        }
    }
}

/// Конвертирует ROI из NV12 в RGB Array3
/// NV12: Y plane (width * height) + UV plane interleaved (width * height / 2)
fn nv12_roi_to_rgb(
    nv12_data: &[u8],
    frame_width: usize,
    frame_height: usize,
    roi: &BBox,
) -> Array3<u8> {
    // Ограничиваем ROI границами кадра
    let x = roi.x.max(0) as usize;
    let y = roi.y.max(0) as usize;
    let w = (roi.width as usize).min(frame_width - x);
    let h = (roi.height as usize).min(frame_height - y);

    let mut rgb = Array3::<u8>::zeros((h, w, 3));

    let y_plane = &nv12_data[..frame_width * frame_height];
    let uv_plane = &nv12_data[frame_width * frame_height..];

    for row in 0..h {
        for col in 0..w {
            let src_x = x + col;
            let src_y = y + row;

            // Y component
            let y_val = y_plane[src_y * frame_width + src_x] as f32;

            // UV components (subsampled 2x2)
            let uv_x = src_x / 2;
            let uv_y = src_y / 2;
            let uv_idx = uv_y * frame_width + uv_x * 2;
            
            let u = uv_plane[uv_idx] as f32 - 128.0;
            let v = uv_plane[uv_idx + 1] as f32 - 128.0;

            // YUV to RGB conversion
            let r = (y_val + 1.402 * v).clamp(0.0, 255.0) as u8;
            let g = (y_val - 0.344 * u - 0.714 * v).clamp(0.0, 255.0) as u8;
            let b = (y_val + 1.772 * u).clamp(0.0, 255.0) as u8;

            rgb[[row, col, 0]] = r;
            rgb[[row, col, 1]] = g;
            rgb[[row, col, 2]] = b;
        }
    }

    rgb
}

// ============================================
// Pipeline
// ============================================

fn create_pipeline(
    device: &str,
    init_config: InitialBBoxConfig,
) -> Result<(gst::Pipeline, Arc<Mutex<FpsAnalyzer>>, Arc<Mutex<TrackerContext>>, Arc<Mutex<TimingStats>>)> {
    gst::init()?;

    let pipeline = gst::Pipeline::new();

    let src = gst::ElementFactory::make("v4l2src")
        .property("device", device)
        .property("do-timestamp", true)
        .build()?;

    let caps_src = gst::ElementFactory::make("capsfilter").build()?;
    caps_src.set_property("caps", &gst::Caps::builder("video/x-raw")
        .field("format", "NV12")
        .field("width", 1920i32)
        .field("height", 1080i32)
        .field("framerate", gst::Fraction::new(60, 1))
        .build());

    // let convert = gst::ElementFactory::make("videoconvert").build()?;

    // let caps_rgb = gst::ElementFactory::make("capsfilter").build()?;
    // caps_rgb.set_property("caps", &gst::Caps::builder("video/x-raw")
    //     .field("format", "RGB")
    //     .build());

    let identity = gst::ElementFactory::make("identity").build()?;

    let convert2 = gst::ElementFactory::make("videoconvert").build()?;

    let sink = gst::ElementFactory::make("autovideosink")
        .property("sync", false)
        .build()?;

    pipeline.add_many([&src, &caps_src, &identity, &convert2, &sink])?;
    gst::Element::link_many([&src, &caps_src, &identity, &convert2, &sink])?;

    let fps_analyzer = Arc::new(Mutex::new(FpsAnalyzer::new(120)));
    let tracker_ctx = Arc::new(Mutex::new(TrackerContext::new(VitTrackConfig::default(), init_config)?));
    let timing_stats = Arc::new(Mutex::new(TimingStats::new(100)));

    let fps_clone = fps_analyzer.clone();
    let tracker_clone = tracker_ctx.clone();
    let timing_clone = timing_stats.clone();

    let frame_counter = Arc::new(AtomicU64::new(0));
    let video_info: Arc<Mutex<Option<gst_video::VideoInfo>>> = Arc::new(Mutex::new(None));
    let video_info_clone = video_info.clone();

    let identity_src_pad = identity.static_pad("src").unwrap();

    identity_src_pad.add_probe(gst::PadProbeType::BUFFER, move |pad, probe_info| {
        let total_start = Instant::now();
        let frame_num = frame_counter.fetch_add(1, Ordering::SeqCst);

        static LAST_FRAME_TIME: std::sync::OnceLock<Mutex<Option<Instant>>> = std::sync::OnceLock::new();
        let last_time_mutex = LAST_FRAME_TIME.get_or_init(|| Mutex::new(None));
        
        let now = Instant::now();
        if let Ok(mut last) = last_time_mutex.lock() {
            if let Some(prev) = *last {
                let interval_ms = now.duration_since(prev).as_secs_f64() * 1000.0;
                // Если интервал > 20ms, значит проблема ДО probe (в videoconvert)
                if frame_num % 60 == 0 {
                    println!("Frame interval: {:.2}ms (max FPS possible: {:.1})", 
                        interval_ms, 1000.0 / interval_ms);

                }
            }
            *last = Some(now);
        }

        // FPS
        if let Ok(mut analyzer) = fps_clone.lock() {
            analyzer.add_frame();
        }

        if let Some(gst::PadProbeData::Buffer(ref mut buffer)) = probe_info.data {
            let mut vi_guard = video_info_clone.lock().unwrap();
            if vi_guard.is_none() {
                if let Some(caps) = pad.current_caps() {
                    if let Ok(info) = gst_video::VideoInfo::from_caps(&caps) {
                        println!("Video: {}x{} {:?} @ {:?}", info.width(), info.height(), info.format(), info.fps());
                        *vi_guard = Some(info);
                    }
                }
            }

            if let Some(ref info) = *vi_guard {
                let buffer_mut = buffer.make_mut();
                if let Ok(mut map) = buffer_mut.map_writable() {
                    let data = map.as_mut_slice();
                    let width = info.width() as usize;
                    let height = info.height() as usize;

                    // КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: используем ArrayView3 без копирования!
                    let copy_start = Instant::now();
                    let frame_view = match ArrayView3::from_shape((height, width, 3), data) {
                        Ok(v) => v,
                        Err(_) => return gst::PadProbeReturn::Ok,
                    };
                    let copy_time_us = copy_start.elapsed().as_micros() as u64;

                    // Трекинг
                    let track_start = Instant::now();
                    let (bbox_result, _track_us, _) = {
                        let mut tracker = tracker_clone.lock().unwrap();
                        tracker.process_frame(&frame_view, frame_num)
                    };
                    let track_time_us = track_start.elapsed().as_micros() as u64;

                    // Получаем данные для отрисовки
                    let (state_name, score) = {
                        let tracker = tracker_clone.lock().unwrap();
                        (tracker.state_name(), tracker.current_score)
                    };

                    // Рисование
                    let draw_start = Instant::now();
                    let fps = fps_clone.lock().map(|a| a.current_fps()).unwrap_or(0.0);

                    // Фон
                    draw_background(data, width, height, 10, 10, 350, 100, 180);

                    // Статус
                    let status_color = match state_name {
                        "TRACKING" => (0, 255, 0),
                        "WAITING" => (255, 255, 0),
                        _ => (255, 0, 0),
                    };
                    draw_text(data, width, height, state_name, 15, 15, status_color, 2);

                    // Статистика
                    let fps_text = format!("FPS: {:.1}", fps);
                    draw_text(data, width, height, &fps_text, 15, 35, (255, 255, 255), 2);

                    let track_text = format!("Track: {:.1}ms", track_time_us as f64 / 1000.0);
                    draw_text(data, width, height, &track_text, 15, 55, (255, 255, 255), 2);

                    let copy_text = format!("Copy: {:.2}ms", copy_time_us as f64 / 1000.0);
                    draw_text(data, width, height, &copy_text, 15, 75, (200, 200, 200), 2);

                    if state_name == "TRACKING" {
                        let score_text = format!("Score: {:.0}%", score * 100.0);
                        draw_text(data, width, height, &score_text, 200, 15, (255, 255, 255), 2);
                    }

                    // BBox
                    if let Some(bbox) = bbox_result {
                        draw_rect(data, width, height, &bbox, (0, 255, 0), 3);
                        let (cx, cy) = (bbox.x + bbox.width / 2, bbox.y + bbox.height / 2);
                        draw_crosshair(data, width, height, cx, cy, 15, (255, 0, 0));
                    }

                    let draw_time_us = draw_start.elapsed().as_micros() as u64;
                    let total_time_us = total_start.elapsed().as_micros() as u64;

                    // Сохраняем статистику
                    if let Ok(mut timing) = timing_clone.lock() {
                        timing.add(copy_time_us, track_time_us, draw_time_us, total_time_us);
                    }

                    // Логирование
                    if frame_num > 0 && frame_num % 120 == 0 {
                        if let Ok(timing) = timing_clone.lock() {
                            println!("[{}] {}", state_name, timing.report());
                        }
                    }
                }
            }
            
        }

        gst::PadProbeReturn::Ok
    });

    Ok((pipeline, fps_analyzer, tracker_ctx, timing_stats))
}

fn main() -> Result<()> {
    println!("==========================================");
    println!("     VitTrack - Optimized");
    println!("==========================================\n");

    let device = "/dev/video11";

    if !std::path::Path::new(device).exists() {
        return Err(anyhow!("Camera {} not found!", device));
    }

    let init_config = InitialBBoxConfig {
        mode: InitMode::DelayedCenter { delay_frames: 60, width: 150, height: 150 },
    };

    let (pipeline, fps_analyzer, tracker_ctx, timing_stats) = create_pipeline(device, init_config)?;

    println!("Starting...");
    println!("Tracker will init after 1 second at screen center");
    println!("Press Ctrl+C to exit\n");

    pipeline.set_state(gst::State::Playing)?;

    let running = Arc::new(AtomicBool::new(true));
    let running_clone = running.clone();

    std::thread::spawn(move || {
        let mut signals = signal_hook::iterator::Signals::new(&[signal_hook::consts::SIGINT]).unwrap();
        for _ in signals.forever() {
            println!("\nShutting down...");
            running_clone.store(false, Ordering::SeqCst);
            break;
        }
    });

    let bus = pipeline.bus().unwrap();

    while running.load(Ordering::SeqCst) {
        if let Some(msg) = bus.timed_pop(gst::ClockTime::from_mseconds(100)) {
            use gst::MessageView;
            match msg.view() {
                MessageView::Error(err) => {
                    eprintln!("Error: {}", err.error());
                    break;
                }
                MessageView::Eos(..) => break,
                MessageView::StateChanged(sc) => {
                    if sc.src().map(|s| s == &pipeline).unwrap_or(false) {
                        println!("Pipeline: {:?} -> {:?}", sc.old(), sc.current());
                    }
                }
                _ => {}
            }
        }
    }

    println!("\n========== FINAL STATISTICS ==========");
    if let Ok(analyzer) = fps_analyzer.lock() {
        println!("FPS: {:.1} (avg: {:.1})", analyzer.current_fps(), analyzer.avg_fps());
        println!("Total frames: {}", analyzer.total_frames);
    }
    if let Ok(tracker) = tracker_ctx.lock() {
        println!("Tracked frames: {}", tracker.total_tracked_frames);
        println!("Lost count: {}", tracker.lost_count);
    }
    if let Ok(timing) = timing_stats.lock() {
        println!("Timing: {}", timing.report());
    }
    println!("=======================================");

    pipeline.set_state(gst::State::Null)?;

    Ok(())
}