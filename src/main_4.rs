use anyhow::{anyhow, Result};
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_video as gst_video;
use ndarray::{Array3, ArrayViewMut3};
use std::collections::VecDeque;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

// ============================================
// Импорт вашего трекера - раскомментируйте
// ============================================
use vit_tracker::{VitTrack, tracker::VitTrackConfig, BBox, TrackingResult};



/// Конфигурация начального bbox
#[derive(Debug, Clone)]
pub struct InitialBBoxConfig {
    /// Способ инициализации
    pub mode: InitMode,
}

#[derive(Debug, Clone)]
pub enum InitMode {
    /// Фиксированный bbox
    Fixed(BBox),
    /// По центру экрана с заданным размером
    Center { width: i32, height: i32 },
    /// Ожидание N кадров, затем центр
    DelayedCenter { delay_frames: u64, width: i32, height: i32 },
}

impl Default for InitialBBoxConfig {
    fn default() -> Self {
        Self {
            mode: InitMode::DelayedCenter {
                delay_frames: 30, // Полсекунды при 60 FPS
                width: 100,
                height: 100,
            },
        }
    }
}

/// Состояние приложения
enum AppState {
    /// Ожидание первого кадра / инициализации
    WaitingForInit { frames_waited: u64 },
    /// Трекинг активен
    Tracking,
    /// Трекинг потерян
    Lost { frames_lost: u64 },
}

/// Контекст трекера
struct TrackerContext {
    tracker: VitTrack,
    state: AppState,
    init_config: InitialBBoxConfig,
    current_bbox: Option<BBox>,
    current_score: f32,
    
    // Статистика
    track_times_us: VecDeque<u64>,
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
            track_times_us: VecDeque::with_capacity(100),
            total_tracked_frames: 0,
            lost_count: 0,
        })
    }

    fn process_frame(&mut self, image: &Array3<u8>, frame_num: u64) -> Option<BBox> {
        let (height, width, _) = (image.shape()[0], image.shape()[1], image.shape()[2]);

        match &mut self.state {
            AppState::WaitingForInit { frames_waited } => {
                *frames_waited += 1;

                // Определяем bbox для инициализации
                let init_bbox = match &self.init_config.mode {
                    InitMode::Fixed(bbox) => {
                        Some(*bbox)
                    }
                    InitMode::Center { width: w, height: h } => {
                        Some(BBox::new(
                            (width as i32 - w) / 2,
                            (height as i32 - h) / 2,
                            *w,
                            *h,
                        ))
                    }
                    InitMode::DelayedCenter { delay_frames, width: w, height: h } => {
                        if *frames_waited >= *delay_frames {
                            Some(BBox::new(
                                (width as i32 - w) / 2,
                                (height as i32 - h) / 2,
                                *w,
                                *h,
                            ))
                        } else {
                            None
                        }
                    }
                };

                if let Some(bbox) = init_bbox {
                    self.tracker.init(image, bbox);
                    self.current_bbox = Some(bbox);
                    self.state = AppState::Tracking;
                    println!("Frame {}: Tracker initialized at {:?}", frame_num, bbox);
                    return Some(bbox);
                }

                None
            }

            AppState::Tracking => {
                let start = Instant::now();
                
                match self.tracker.update(image) {
                    Ok(result) => {
                        let elapsed_us = start.elapsed().as_micros() as u64;
                        self.record_track_time(elapsed_us);

                        if result.success && result.score > 0.3 {
                            let bbox = BBox::from_array(&result.bbox);
                            self.current_bbox = Some(bbox);
                            self.current_score = result.score;
                            self.total_tracked_frames += 1;
                            Some(bbox)
                        } else {
                            // Трекинг потерян
                            self.state = AppState::Lost { frames_lost: 0 };
                            self.lost_count += 1;
                            println!("Frame {}: Track lost (score: {:.2})", frame_num, result.score);
                            None
                        }
                    }
                    Err(_) => {
                        self.state = AppState::Lost { frames_lost: 0 };
                        self.lost_count += 1;
                        None
                    }
                }
            }

            AppState::Lost { frames_lost } => {
                *frames_lost += 1;
                
                // Можно добавить логику переинициализации
                // Например, через N кадров попробовать снова
                if *frames_lost > 60 {
                    // Сброс в начальное состояние
                    self.state = AppState::WaitingForInit { frames_waited: 0 };
                    println!("Frame {}: Resetting tracker after {} lost frames", frame_num, frames_lost);
                }
                
                None
            }
        }
    }

    fn record_track_time(&mut self, time_us: u64) {
        if self.track_times_us.len() >= 100 {
            self.track_times_us.pop_front();
        }
        self.track_times_us.push_back(time_us);
    }

    fn avg_track_time_ms(&self) -> f64 {
        if self.track_times_us.is_empty() {
            return 0.0;
        }
        let sum: u64 = self.track_times_us.iter().sum();
        (sum as f64 / self.track_times_us.len() as f64) / 1000.0
    }

    fn state_name(&self) -> &'static str {
        match self.state {
            AppState::WaitingForInit { .. } => "WAITING",
            AppState::Tracking => "TRACKING",
            AppState::Lost { .. } => "LOST",
        }
    }
}

/// FPS анализатор
struct FpsAnalyzer {
    frame_times: VecDeque<Instant>,
    intervals_us: VecDeque<u64>,
    max_samples: usize,
    total_frames: u64,
    start_time: Instant,
}

impl FpsAnalyzer {
    fn new(max_samples: usize) -> Self {
        Self {
            frame_times: VecDeque::with_capacity(max_samples),
            intervals_us: VecDeque::with_capacity(max_samples),
            max_samples,
            total_frames: 0,
            start_time: Instant::now(),
        }
    }

    fn add_frame(&mut self, time: Instant) {
        if let Some(&last_time) = self.frame_times.back() {
            let interval_us = time.duration_since(last_time).as_micros() as u64;
            if self.intervals_us.len() >= self.max_samples {
                self.intervals_us.pop_front();
            }
            self.intervals_us.push_back(interval_us);
        }

        if self.frame_times.len() >= self.max_samples {
            self.frame_times.pop_front();
        }
        self.frame_times.push_back(time);
        self.total_frames += 1;
    }

    fn current_fps(&self) -> f64 {
        if self.intervals_us.is_empty() {
            return 0.0;
        }
        let avg_us = self.intervals_us.iter().sum::<u64>() as f64 / self.intervals_us.len() as f64;
        if avg_us > 0.0 { 1_000_000.0 / avg_us } else { 0.0 }
    }

    fn avg_fps(&self) -> f64 {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        if elapsed > 0.0 { self.total_frames as f64 / elapsed } else { 0.0 }
    }
}

// ============================================
// Рисование
// ============================================

fn draw_rect(
    buffer: &mut [u8],
    width: usize,
    height: usize,
    bbox: &BBox,
    color: (u8, u8, u8),
    thickness: usize,
) {
    let x1 = (bbox.x.max(0) as usize).min(width.saturating_sub(1));
    let y1 = (bbox.y.max(0) as usize).min(height.saturating_sub(1));
    let x2 = ((bbox.x + bbox.width).max(0) as usize).min(width.saturating_sub(1));
    let y2 = ((bbox.y + bbox.height).max(0) as usize).min(height.saturating_sub(1));

    // Горизонтальные линии
    for t in 0..thickness {
        // Верхняя
        let y = y1 + t;
        if y < height {
            for x in x1..=x2 {
                set_pixel(buffer, width, x, y, color);
            }
        }
        // Нижняя
        if y2 >= t {
            let y = y2 - t;
            if y < height {
                for x in x1..=x2 {
                    set_pixel(buffer, width, x, y, color);
                }
            }
        }
    }

    // Вертикальные линии
    for t in 0..thickness {
        // Левая
        let x = x1 + t;
        if x < width {
            for y in y1..=y2 {
                set_pixel(buffer, width, x, y, color);
            }
        }
        // Правая
        if x2 >= t {
            let x = x2 - t;
            if x < width {
                for y in y1..=y2 {
                    set_pixel(buffer, width, x, y, color);
                }
            }
        }
    }
}

fn draw_crosshair(
    buffer: &mut [u8],
    width: usize,
    height: usize,
    cx: i32,
    cy: i32,
    size: i32,
    color: (u8, u8, u8),
) {
    let cx = cx.max(0) as usize;
    let cy = cy.max(0) as usize;
    let size = size as usize;

    // Горизонтальная
    if cy < height {
        for x in cx.saturating_sub(size)..=(cx + size).min(width - 1) {
            set_pixel(buffer, width, x, cy, color);
        }
    }

    // Вертикальная
    if cx < width {
        for y in cy.saturating_sub(size)..=(cy + size).min(height - 1) {
            set_pixel(buffer, width, cx, y, color);
        }
    }
}

#[inline]
fn set_pixel(buffer: &mut [u8], width: usize, x: usize, y: usize, color: (u8, u8, u8)) {
    let idx = (y * width + x) * 3;
    if idx + 2 < buffer.len() {
        buffer[idx] = color.0;
        buffer[idx + 1] = color.1;
        buffer[idx + 2] = color.2;
    }
}

fn draw_text(
    buffer: &mut [u8],
    width: usize,
    height: usize,
    text: &str,
    x: usize,
    y: usize,
    color: (u8, u8, u8),
    scale: usize,
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
        ("X", [0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b01010, 0b10001]),
        ("x", [0b00000, 0b00000, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001]),
        ("m", [0b00000, 0b00000, 0b11010, 0b10101, 0b10101, 0b10001, 0b10001]),
        ("s", [0b00000, 0b00000, 0b01110, 0b10000, 0b01110, 0b00001, 0b11110]),
        ("c", [0b00000, 0b00000, 0b01110, 0b10000, 0b10000, 0b10001, 0b01110]),
        ("o", [0b00000, 0b00000, 0b01110, 0b10001, 0b10001, 0b10001, 0b01110]),
        ("r", [0b00000, 0b00000, 0b10110, 0b11001, 0b10000, 0b10000, 0b10000]),
        ("e", [0b00000, 0b00000, 0b01110, 0b10001, 0b11111, 0b10000, 0b01110]),
        ("[", [0b01110, 0b01000, 0b01000, 0b01000, 0b01000, 0b01000, 0b01110]),
        ("]", [0b01110, 0b00010, 0b00010, 0b00010, 0b00010, 0b00010, 0b01110]),
        ("%", [0b11001, 0b11010, 0b00100, 0b00100, 0b01000, 0b01011, 0b10011]),
    ];

    let mut cursor_x = x;

    for ch in text.chars() {
        let glyph = FONT
            .iter()
            .find(|(c, _)| c.chars().next() == Some(ch))
            .map(|(_, g)| g);

        if let Some(glyph) = glyph {
            for (row, &bits) in glyph.iter().enumerate() {
                for col in 0..5 {
                    if (bits >> (4 - col)) & 1 == 1 {
                        for dy in 0..scale {
                            for dx in 0..scale {
                                let px = cursor_x + col * scale + dx;
                                let py = y + row * scale + dy;
                                if px < width && py < height {
                                    set_pixel(buffer, width, px, py, color);
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

/// Рисует полупрозрачный фон для текста
fn draw_text_background(
    buffer: &mut [u8],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    text_width: usize,
    text_height: usize,
    opacity: u8,
) {
    for py in y..(y + text_height).min(height) {
        for px in x..(x + text_width).min(width) {
            let idx = (py * width + px) * 3;
            if idx + 2 < buffer.len() {
                // Затемняем фон
                buffer[idx] = (buffer[idx] as u16 * (255 - opacity as u16) / 255) as u8;
                buffer[idx + 1] = (buffer[idx + 1] as u16 * (255 - opacity as u16) / 255) as u8;
                buffer[idx + 2] = (buffer[idx + 2] as u16 * (255 - opacity as u16) / 255) as u8;
            }
        }
    }
}

// ============================================
// GStreamer Pipeline
// ============================================

fn create_pipeline(
    device: &str,
    init_config: InitialBBoxConfig,
) -> Result<(gst::Pipeline, Arc<Mutex<FpsAnalyzer>>, Arc<Mutex<TrackerContext>>)> {
    gst::init()?;

    let pipeline = gst::Pipeline::new();

    // Источник
    let src = gst::ElementFactory::make("v4l2src")
        .property("device", device)
        .property("do-timestamp", true)
        .build()?;

    // Входной формат камеры - 60 FPS, NV12
    let caps_src = gst::ElementFactory::make("capsfilter").build()?;
    caps_src.set_property(
        "caps",
        &gst::Caps::builder("video/x-raw")
            .field("format", "NV12")
            .field("width", 1920i32)
            .field("height", 1080i32)
            .field("framerate", gst::Fraction::new(60, 1))
            .build(),
    );

    // Конвертер в RGB для трекера
    let convert = gst::ElementFactory::make("videoconvert").build()?;

    // RGB формат
    let caps_rgb = gst::ElementFactory::make("capsfilter").build()?;
    caps_rgb.set_property(
        "caps",
        &gst::Caps::builder("video/x-raw")
            .field("format", "RGB")
            .build(),
    );

    let identity = gst::ElementFactory::make("identity").build()?;

    // Конвертер для вывода
    let convert2 = gst::ElementFactory::make("videoconvert").build()?;

    let sink = gst::ElementFactory::make("autovideosink")
        .property("sync", false)
        .build()?;

    pipeline.add_many([&src, &caps_src, &convert, &caps_rgb, &identity, &convert2, &sink])?;
    gst::Element::link_many([&src, &caps_src, &convert, &caps_rgb, &identity, &convert2, &sink])?;

    // Контексты
    let fps_analyzer = Arc::new(Mutex::new(FpsAnalyzer::new(120)));
    let tracker_ctx = Arc::new(Mutex::new(TrackerContext::new(
        VitTrackConfig::default(),
        init_config,
    )?));

    let fps_clone = fps_analyzer.clone();
    let tracker_clone = tracker_ctx.clone();

    let frame_counter = Arc::new(AtomicU64::new(0));
    let video_info: Arc<Mutex<Option<gst_video::VideoInfo>>> = Arc::new(Mutex::new(None));
    let video_info_clone = video_info.clone();

    let identity_src_pad = identity.static_pad("src").unwrap();

    identity_src_pad.add_probe(gst::PadProbeType::BUFFER, move |pad, probe_info| {
        let capture_time = Instant::now();
        let frame_num = frame_counter.fetch_add(1, Ordering::SeqCst);

        // FPS
        if let Ok(mut analyzer) = fps_clone.lock() {
            analyzer.add_frame(capture_time);
        }

        // Логирование каждые 2 секунды
        if frame_num > 0 && frame_num % 120 == 0 {
            if let (Ok(analyzer), Ok(tracker)) = (fps_clone.lock(), tracker_clone.lock()) {
                println!(
                    "[{}] FPS: {:.1} (avg: {:.1}) | Track: {:.2}ms | Score: {:.2} | Lost: {}",
                    tracker.state_name(),
                    analyzer.current_fps(),
                    analyzer.avg_fps(),
                    tracker.avg_track_time_ms(),
                    tracker.current_score,
                    tracker.lost_count,
                );
            }
        }

        if let Some(gst::PadProbeData::Buffer(ref mut buffer)) = probe_info.data {
            // Video info
            let mut vi_guard = video_info_clone.lock().unwrap();
            if vi_guard.is_none() {
                if let Some(caps) = pad.current_caps() {
                    if let Ok(info) = gst_video::VideoInfo::from_caps(&caps) {
                        println!(
                            "Video: {}x{} {:?} @ {:?}",
                            info.width(),
                            info.height(),
                            info.format(),
                            info.fps()
                        );
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

                    // Создаём Array3 для трекера
                    if let Ok(frame_view) =
                        ArrayViewMut3::from_shape((height, width, 3), data)
                    {
                        let frame_array: Array3<u8> = frame_view.to_owned();

                        // Обрабатываем кадр трекером
                        let mut tracker = tracker_clone.lock().unwrap();
                        let bbox_result = tracker.process_frame(&frame_array, frame_num);
                        let state_name = tracker.state_name();
                        let score = tracker.current_score;
                        let track_time = tracker.avg_track_time_ms();
                        drop(tracker); // Освобождаем lock

                        // Рисуем результат
                        let fps = fps_clone.lock().map(|a| a.current_fps()).unwrap_or(0.0);

                        // Фон для текста
                        draw_text_background(data, width, height, 10, 10, 300, 80, 180);

                        // Статус
                        let status_color = match state_name {
                            "TRACKING" => (0, 255, 0),
                            "WAITING" => (255, 255, 0),
                            "LOST" => (255, 0, 0),
                            _ => (255, 255, 255),
                        };
                        draw_text(data, width, height, state_name, 15, 15, status_color, 2);

                        // FPS
                        let fps_text = format!("FPS: {:.1}", fps);
                        draw_text(data, width, height, &fps_text, 15, 35, (255, 255, 255), 2);

                        // Track time
                        let time_text = format!("Track: {:.1}ms", track_time);
                        draw_text(data, width, height, &time_text, 15, 55, (255, 255, 255), 2);

                        // Score
                        if state_name == "TRACKING" {
                            let score_text = format!("Score: {:.0}%", score * 100.0);
                            draw_text(data, width, height, &score_text, 15, 75, (255, 255, 255), 2);
                        }

                        // Рисуем bbox если есть
                        if let Some(bbox) = bbox_result {
                            // Основной прямоугольник
                            draw_rect(data, width, height, &bbox, (0, 255, 0), 3);

                            // Центр объекта
                            let (cx, cy) = (bbox.x + bbox.width / 2, bbox.y + bbox.height / 2);
                            draw_crosshair(data, width, height, cx, cy, 15, (255, 0, 0));

                            // Координаты bbox
                            let coord_text = format!(
                                "[{}.{} {}x{}]",
                                bbox.x, bbox.y, bbox.width, bbox.height
                            );
                            let text_y = (bbox.y - 20).max(0) as usize;
                            draw_text(
                                data,
                                width,
                                height,
                                &coord_text,
                                bbox.x.max(0) as usize,
                                text_y,
                                (0, 255, 0),
                                1,
                            );
                        }
                    }
                }
                
            }
        }

        gst::PadProbeReturn::Ok
    });

    Ok((pipeline, fps_analyzer, tracker_ctx))
}

// ============================================
// Main
// ============================================

fn main() -> Result<()> {
    println!("==========================================");
    println!("     VitTrack Object Tracker");
    println!("==========================================\n");

    let device = "/dev/video11";

    if !std::path::Path::new(device).exists() {
        return Err(anyhow!("Камера {} не найдена!", device));
    }

    // Конфигурация начального bbox
    // Вариант 1: Фиксированный bbox
    // let init_config = InitialBBoxConfig {
    //     mode: InitMode::Fixed(BBox::new(800, 400, 150, 150)),
    // };

    // Вариант 2: По центру с задержкой
    let init_config = InitialBBoxConfig {
        mode: InitMode::DelayedCenter {
            delay_frames: 120, // 1 секунда при 60 FPS
            width: 128,
            height: 128,
        },
    };
    // Вариант 3: Сразу по центру
    // let init_config = InitialBBoxConfig {
    //     mode: InitMode::Center { width: 150, height: 150 },
    // };

    let (pipeline, fps_analyzer, tracker_ctx) = create_pipeline(device, init_config)?;

    println!("Запуск...");
    println!("Трекер инициализируется через 1 секунду по центру экрана");
    println!("Нажмите Ctrl+C для выхода\n");

    pipeline.set_state(gst::State::Playing)?;

    // Ctrl+C
    let running = Arc::new(AtomicBool::new(true));
    let running_clone = running.clone();

    std::thread::spawn(move || {
        let mut signals = signal_hook::iterator::Signals::new(&[signal_hook::consts::SIGINT])
            .expect("Ошибка сигналов");
        for _ in signals.forever() {
            println!("\nЗавершение...");
            running_clone.store(false, Ordering::SeqCst);
            break;
        }
    });

    // Основной цикл
    let bus = pipeline.bus().unwrap();

    while running.load(Ordering::SeqCst) {
        if let Some(msg) = bus.timed_pop(gst::ClockTime::from_mseconds(100)) {
            use gst::MessageView;
            match msg.view() {
                MessageView::Error(err) => {
                    eprintln!("Ошибка: {}", err.error());
                    if let Some(debug) = err.debug() {
                        eprintln!("Debug: {}", debug);
                    }
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

    // Финальная статистика
    println!("\n========== STATISTICS ==========");
    if let Ok(analyzer) = fps_analyzer.lock() {
        println!("FPS: {:.1} (avg: {:.1})", analyzer.current_fps(), analyzer.avg_fps());
        println!("Total frames: {}", analyzer.total_frames);
    }
    if let Ok(tracker) = tracker_ctx.lock() {
        println!("Tracked frames: {}", tracker.total_tracked_frames);
        println!("Lost count: {}", tracker.lost_count);
        println!("Avg track time: {:.2}ms", tracker.avg_track_time_ms());
    }
    println!("================================");

    pipeline.set_state(gst::State::Null)?;

    Ok(())
}