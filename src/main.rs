mod nv12_convert;
mod rga;

use anyhow::{anyhow, Result};
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_video as gst_video;
use ndarray::{Array3, ArrayView3};
use nv12_convert::*;
use std::collections::VecDeque;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use vit_tracker::{BBox, VitTrack};

const MODEL_PATH: &str = "/home/radxa/repos/rust/vit_tracker/models/object_tracking_vittrack_2023sep.rknn";

// ============================================
// Статистика
// ============================================

struct TimingStats {
    convert_us: VecDeque<u64>,
    track_us: VecDeque<u64>,
    draw_us: VecDeque<u64>,
    frame_intervals_us: VecDeque<u64>,
    max_samples: usize,
}

impl TimingStats {
    fn new(max_samples: usize) -> Self {
        Self {
            convert_us: VecDeque::with_capacity(max_samples),
            track_us: VecDeque::with_capacity(max_samples),
            draw_us: VecDeque::with_capacity(max_samples),
            frame_intervals_us: VecDeque::with_capacity(max_samples),
            max_samples,
        }
    }

    fn push(queue: &mut VecDeque<u64>, value: u64, max_samples: usize) {
        if queue.len() >= max_samples {
            queue.pop_front();
        }
        queue.push_back(value);
    }

    fn add(&mut self, convert: u64, track: u64, draw: u64) {
        Self::push(&mut self.convert_us, convert, self.max_samples);
        Self::push(&mut self.track_us, track, self.max_samples);
        Self::push(&mut self.draw_us, draw, self.max_samples);
    }

    fn add_frame_interval(&mut self, interval: u64) {
        Self::push(&mut self.frame_intervals_us, interval, self.max_samples);
    }

    fn avg(queue: &VecDeque<u64>) -> f64 {
        if queue.is_empty() {
            0.0
        } else {
            queue.iter().sum::<u64>() as f64 / queue.len() as f64 / 1000.0
        }
    }

    fn input_fps(&self) -> f64 {
        if self.frame_intervals_us.is_empty() {
            return 0.0;
        }
        let avg_us = self.frame_intervals_us.iter().sum::<u64>() as f64
            / self.frame_intervals_us.len() as f64;
        if avg_us > 0.0 {
            1_000_000.0 / avg_us
        } else {
            0.0
        }
    }

    fn report(&self) -> String {
        format!(
            "FPS: {:.1} | conv: {:.1}ms | track: {:.1}ms | draw: {:.1}ms",
            self.input_fps(),
            Self::avg(&self.convert_us),
            Self::avg(&self.track_us),
            Self::avg(&self.draw_us),
        )
    }
}

// ============================================
// Состояние
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

// ============================================
// Контекст трекера
// ============================================

struct TrackerContext {
    tracker: VitTrack,
    state: AppState,
    delay_frames: u64,
    init_bbox_size: (i32, i32),
    current_bbox: Option<BBox>,
    current_score: f32,
    total_tracked: u64,
    lost_count: u64,
    // Сохраняем кадр инициализации для сравнения
    init_frame_pixels: Option<[u8; 9]>, // 3 пикселя по 3 канала
    frames_since_init: u64,
}

impl TrackerContext {
    fn new(model_path: &str, delay_frames: u64, init_width: i32, init_height: i32) -> Result<Self> {
        println!("Loading tracker model: {}", model_path);
        let tracker = VitTrack::new(model_path)
            .map_err(|e| anyhow!("Failed to create tracker: {:?}", e))?;
        println!("Tracker loaded");

        Ok(Self {
            tracker,
            state: AppState::WaitingForInit { frames_waited: 0 },
            delay_frames,
            init_bbox_size: (init_width, init_height),
            current_bbox: None,
            current_score: 0.0,
            total_tracked: 0,
            lost_count: 0,
            init_frame_pixels: None,
            frames_since_init: 0,
        })
    }

    fn process_frame(
        &mut self,
        full_image: &ArrayView3<u8>,
        frame_num: u64,
    ) -> Option<BBox> {
        let current_state = self.state;
        let (img_h, img_w, _) = full_image.dim();

        match current_state {
            AppState::WaitingForInit { frames_waited } => {
                let new_waited = frames_waited + 1;

                if new_waited >= self.delay_frames {
                    let (bbox_w, bbox_h) = self.init_bbox_size;

                    let bbox = BBox::new(
                        (img_w as i32 - bbox_w) / 2,
                        (img_h as i32 - bbox_h) / 2,
                        bbox_w,
                        bbox_h,
                    );

                    // Сохраняем пиксели для диагностики
                    let center_y = img_h / 2;
                    let center_x = img_w / 2;
                    self.init_frame_pixels = Some([
                        full_image[[center_y, center_x, 0]],
                        full_image[[center_y, center_x, 1]],
                        full_image[[center_y, center_x, 2]],
                        full_image[[0, 0, 0]],
                        full_image[[0, 0, 1]],
                        full_image[[0, 0, 2]],
                        full_image[[bbox.y as usize, bbox.x as usize, 0]],
                        full_image[[bbox.y as usize, bbox.x as usize, 1]],
                        full_image[[bbox.y as usize, bbox.x as usize, 2]],
                    ]);

                    println!("\n========== INIT ==========");
                    println!("Frame {}: Initializing tracker", frame_num);
                    println!("Image size: {}x{}", img_w, img_h);
                    println!("BBox: x={}, y={}, w={}, h={}", bbox.x, bbox.y, bbox.width, bbox.height);
                    println!("Pixel[center] BGR: [{}, {}, {}]",
                             full_image[[center_y, center_x, 0]],
                             full_image[[center_y, center_x, 1]],
                             full_image[[center_y, center_x, 2]]);
                    println!("Pixel[0,0] BGR: [{}, {}, {}]",
                             full_image[[0, 0, 0]], full_image[[0, 0, 1]], full_image[[0, 0, 2]]);
                    println!("Pixel[bbox corner] BGR: [{}, {}, {}]",
                             full_image[[bbox.y as usize, bbox.x as usize, 0]],
                             full_image[[bbox.y as usize, bbox.x as usize, 1]],
                             full_image[[bbox.y as usize, bbox.x as usize, 2]]);
                    println!("==============================\n");

                    self.tracker.init(full_image, bbox);

                    self.current_bbox = Some(bbox);
                    self.state = AppState::Tracking;
                    self.frames_since_init = 0;
                    return Some(bbox);
                }

                self.state = AppState::WaitingForInit { frames_waited: new_waited };
                None
            }

            AppState::Tracking => {
                self.frames_since_init += 1;

                // Сравниваем пиксели с кадром инициализации (первые 5 кадров)
                if self.frames_since_init <= 5 {
                    let (img_h, img_w, _) = full_image.dim();
                    let center_y = img_h / 2;
                    let center_x = img_w / 2;

                    if let Some(init_pixels) = &self.init_frame_pixels {
                        let current_center = [
                            full_image[[center_y, center_x, 0]],
                            full_image[[center_y, center_x, 1]],
                            full_image[[center_y, center_x, 2]],
                        ];

                        let diff: i32 = (0..3)
                            .map(|i| (init_pixels[i] as i32 - current_center[i] as i32).abs())
                            .sum();

                        println!("Frame +{}: center pixel diff = {} (init: [{},{},{}] curr: [{},{},{}])",
                                 self.frames_since_init, diff,
                                 init_pixels[0], init_pixels[1], init_pixels[2],
                                 current_center[0], current_center[1], current_center[2]);
                    }
                }

                match self.tracker.update(full_image) {
                    Ok(result) => {
                        // Логируем первые 10 кадров после init
                        if self.frames_since_init <= 10 {
                            println!("  Frame +{}: success={}, score={:.3}, bbox=[{}, {}, {}, {}]",
                                     self.frames_since_init,
                                     result.success, result.score,
                                     result.bbox[0], result.bbox[1], result.bbox[2], result.bbox[3]);
                        }

                        if result.success && result.score > 0.3 {
                            let bbox = BBox::from_array(&result.bbox);
                            self.current_bbox = Some(bbox);
                            self.current_score = result.score;
                            self.total_tracked += 1;
                            Some(bbox)
                        } else {
                            println!("\n!!! LOST at frame +{} (score={:.3}) !!!\n",
                                     self.frames_since_init, result.score);
                            self.state = AppState::Lost { frames_lost: 0 };
                            self.lost_count += 1;
                            self.current_score = 0.0;
                            None
                        }
                    }
                    Err(e) => {
                        println!("Frame {}: Error: {:?}", frame_num, e);
                        self.state = AppState::Lost { frames_lost: 0 };
                        self.lost_count += 1;
                        None
                    }
                }
            }

            AppState::Lost { frames_lost } => {
                let new_lost = frames_lost + 1;
                if new_lost > 120 {
                    println!("Resetting tracker...");
                    self.current_bbox = None;
                    self.state = AppState::WaitingForInit { frames_waited: 0 };
                } else {
                    self.state = AppState::Lost { frames_lost: new_lost };
                }
                None
            }
        }
    }
}

// ============================================
// Конвертация NV12 -> BGR/RGB с разными вариантами
// ============================================

/// Попробуем разные варианты конвертации
fn nv12_to_bgr_v1(nv12_data: &[u8], width: usize, height: usize) -> Array3<u8> {
    // Стандартный NV12: Y plane, потом UV interleaved
    nv12_full_to_bgr(nv12_data, width, height)
}

fn nv12_to_rgb(nv12_data: &[u8], width: usize, height: usize) -> Array3<u8> {
    nv12_roi_to_array(nv12_data, width, height, 0, 0, width, height, ColorOrder::RGB)
}

/// NV21 вместо NV12 (V и U поменяны местами)
fn nv21_to_bgr(nv12_data: &[u8], width: usize, height: usize) -> Array3<u8> {
    let mut result = Array3::<u8>::zeros((height, width, 3));

    let y_plane_size = width * height;
    if nv12_data.len() < y_plane_size * 3 / 2 {
        return result;
    }

    let y_plane = &nv12_data[..y_plane_size];
    let uv_plane = &nv12_data[y_plane_size..];

    for row in 0..height {
        for col in 0..width {
            let y_val = y_plane[row * width + col] as i32;

            let uv_x = col / 2;
            let uv_y = row / 2;
            let uv_idx = uv_y * width + uv_x * 2;

            // NV21: VU вместо UV
            let v = uv_plane.get(uv_idx).copied().unwrap_or(128) as i32 - 128;
            let u = uv_plane.get(uv_idx + 1).copied().unwrap_or(128) as i32 - 128;

            let c = y_val - 16;
            let r = ((298 * c + 409 * v + 128) >> 8).clamp(0, 255) as u8;
            let g = ((298 * c - 100 * u - 208 * v + 128) >> 8).clamp(0, 255) as u8;
            let b = ((298 * c + 516 * u + 128) >> 8).clamp(0, 255) as u8;

            result[[row, col, 0]] = b;
            result[[row, col, 1]] = g;
            result[[row, col, 2]] = r;
        }
    }

    result
}

// ============================================
// Pipeline
// ============================================

fn create_pipeline(
    device: &str,
) -> Result<(gst::Pipeline, Arc<Mutex<TrackerContext>>, Arc<Mutex<TimingStats>>)> {
    gst::init()?;

    let pipeline = gst::Pipeline::new();

    let src = gst::ElementFactory::make("v4l2src")
        .property("device", device)
        .property("do-timestamp", true)
        .build()?;

    let caps_nv12 = gst::ElementFactory::make("capsfilter").build()?;
    caps_nv12.set_property(
        "caps",
        &gst::Caps::builder("video/x-raw")
            .field("format", "NV12")
            .field("width", 1920i32)
            .field("height", 1080i32)
            .field("framerate", gst::Fraction::new(60, 1))
            .build(),
    );

    let identity = gst::ElementFactory::make("identity").build()?;
    let convert = gst::ElementFactory::make("videoconvert").build()?;
    let sink = gst::ElementFactory::make("autovideosink")
        .property("sync", false)
        .build()?;

    pipeline.add_many([&src, &caps_nv12, &identity, &convert, &sink])?;
    gst::Element::link_many([&src, &caps_nv12, &identity, &convert, &sink])?;

    let tracker_ctx = Arc::new(Mutex::new(TrackerContext::new(
        MODEL_PATH,
        60,
        150,
        150,
    )?));
    let timing_stats = Arc::new(Mutex::new(TimingStats::new(120)));

    let tracker_clone = tracker_ctx.clone();
    let timing_clone = timing_stats.clone();

    let frame_counter = Arc::new(AtomicU64::new(0));
    let video_info: Arc<Mutex<Option<gst_video::VideoInfo>>> = Arc::new(Mutex::new(None));
    let video_info_clone = video_info.clone();
    let last_frame_time: Arc<Mutex<Option<Instant>>> = Arc::new(Mutex::new(None));
    let last_frame_time_clone = last_frame_time.clone();

    // Попробуем разные варианты конвертации
    let conversion_mode = Arc::new(AtomicU64::new(0)); // 0=BGR, 1=RGB, 2=NV21
    let conversion_mode_clone = conversion_mode.clone();

    let identity_src_pad = identity.static_pad("src").unwrap();

    identity_src_pad.add_probe(gst::PadProbeType::BUFFER, move |pad, probe_info| {
        let now = Instant::now();
        {
            let mut last = last_frame_time_clone.lock().unwrap();
            if let Some(prev) = *last {
                let interval = now.duration_since(prev).as_micros() as u64;
                if let Ok(mut timing) = timing_clone.lock() {
                    timing.add_frame_interval(interval);
                }
            }
            *last = Some(now);
        }

        let frame_num = frame_counter.fetch_add(1, Ordering::SeqCst);

        let buffer = match &mut probe_info.data {
            Some(gst::PadProbeData::Buffer(buffer)) => buffer,
            _ => return gst::PadProbeReturn::Ok,
        };

        let mut vi_guard = video_info_clone.lock().unwrap();
        if vi_guard.is_none() {
            if let Some(caps) = pad.current_caps() {
                if let Ok(info) = gst_video::VideoInfo::from_caps(&caps) {
                    println!(
                        "Video: {}x{} {:?} @ {:?}",
                        info.width(), info.height(), info.format(), info.fps()
                    );
                    *vi_guard = Some(info);
                }
            }
        }

        let info = match *vi_guard {
            Some(ref info) => info,
            None => return gst::PadProbeReturn::Ok,
        };

        let width = info.width() as usize;
        let height = info.height() as usize;

        let buffer = buffer.make_mut();
        let Ok(mut map) = buffer.map_writable() else {
            return gst::PadProbeReturn::Ok;
        };

        let nv12_data = map.as_mut_slice();

        // Выбираем режим конвертации
        let mode = conversion_mode_clone.load(Ordering::SeqCst);
        let convert_start = Instant::now();

        let image: Array3<u8> = match mode {
            0 => nv12_to_bgr_v1(nv12_data, width, height),
            1 => nv12_to_rgb(nv12_data, width, height),
            2 => nv21_to_bgr(nv12_data, width, height),
            _ => nv12_to_bgr_v1(nv12_data, width, height),
        };

        let convert_time = convert_start.elapsed().as_micros() as u64;

        // Трекинг
        let track_start = Instant::now();
        let image_view: ArrayView3<u8> = image.view();

        let (bbox_result, state_name, score) = {
            let mut tracker = tracker_clone.lock().unwrap();
            let result = tracker.process_frame(&image_view, frame_num);
            (result, tracker.state.name().to_string(), tracker.current_score)
        };
        let track_time = track_start.elapsed().as_micros() as u64;

        // Отрисовка
        let draw_start = Instant::now();

        draw_background_nv12(nv12_data, width, height, 10, 10, 400, 100, 180);
        draw_text_nv12(nv12_data, width, height, &state_name, 15, 15, 2, 255);

        // Показываем режим конвертации
        let mode_text = match mode {
            0 => "NV12->BGR",
            1 => "NV12->RGB",
            2 => "NV21->BGR",
            _ => "???",
        };
        draw_text_nv12(nv12_data, width, height, mode_text, 15, 85, 1, 200);

        if let Ok(timing) = timing_clone.lock() {
            let fps_text = format!("FPS: {:.1}", timing.input_fps());
            draw_text_nv12(nv12_data, width, height, &fps_text, 15, 40, 2, 255);

            let timing_text = format!(
                "conv:{:.0}ms trk:{:.0}ms",
                convert_time as f64 / 1000.0,
                track_time as f64 / 1000.0
            );
            draw_text_nv12(nv12_data, width, height, &timing_text, 15, 65, 1, 200);
        }

        if state_name == "TRACKING" {
            let score_text = format!("score: {:.0}%", score * 100.0);
            draw_text_nv12(nv12_data, width, height, &score_text, 220, 15, 2, 255);
        }

        if let Some(bbox) = bbox_result {
            draw_rect_nv12(
                nv12_data, width, height,
                bbox.x, bbox.y, bbox.width, bbox.height,
                3, 255,
            );

            let (cx, cy) = bbox.center();
            draw_crosshair_nv12(nv12_data, width, height, cx, cy, 15, 255);
        }

        let draw_time = draw_start.elapsed().as_micros() as u64;

        if let Ok(mut timing) = timing_clone.lock() {
            timing.add(convert_time, track_time, draw_time);
        }

        if frame_num > 0 && frame_num % 120 == 0 {
            if let Ok(timing) = timing_clone.lock() {
                println!("[{}] {}", state_name, timing.report());
            }
        }

        gst::PadProbeReturn::Ok
    });

    Ok((pipeline, tracker_ctx, timing_stats))
}

fn main() -> Result<()> {
    println!("==========================================");
    println!("   VitTrack - Diagnostic Mode");
    println!("==========================================\n");

    let device = "/dev/video11";

    if !std::path::Path::new(device).exists() {
        return Err(anyhow!("Camera {} not found!", device));
    }

    if !Path::new(MODEL_PATH).exists() {
        return Err(anyhow!("Model not found: {}", MODEL_PATH));
    }

    let (pipeline, tracker_ctx, timing_stats) = create_pipeline(device)?;

    println!("Starting with NV12->BGR conversion...");
    println!("Watch the console for diagnostic info");
    println!("Press Ctrl+C to exit\n");

    pipeline.set_state(gst::State::Playing)?;

    let running = Arc::new(AtomicBool::new(true));
    let running_clone = running.clone();

    std::thread::spawn(move || {
        let mut signals =
            signal_hook::iterator::Signals::new(&[signal_hook::consts::SIGINT]).unwrap();
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

    println!("\n========== STATISTICS ==========");
    if let Ok(tracker) = tracker_ctx.lock() {
        println!("Tracked: {}", tracker.total_tracked);
        println!("Lost: {}", tracker.lost_count);
    }
    if let Ok(timing) = timing_stats.lock() {
        println!("{}", timing.report());
    }
    println!("=================================");

    pipeline.set_state(gst::State::Null)?;

    Ok(())
}