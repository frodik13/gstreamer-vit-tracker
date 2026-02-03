use anyhow::{anyhow, Result};
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_video as gst_video;
use ndarray::ArrayView3;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Анализатор FPS
struct FpsAnalyzer {
    frame_times: VecDeque<Instant>,
    intervals_us: VecDeque<u64>,
    max_samples: usize,
    total_frames: u64,
    start_time: Instant,
    min_interval_us: u64,
    max_interval_us: u64,
}

impl FpsAnalyzer {
    fn new(max_samples: usize) -> Self {
        Self {
            frame_times: VecDeque::with_capacity(max_samples),
            intervals_us: VecDeque::with_capacity(max_samples),
            max_samples,
            total_frames: 0,
            start_time: Instant::now(),
            min_interval_us: u64::MAX,
            max_interval_us: 0,
        }
    }

    fn add_frame(&mut self, time: Instant) {
        if let Some(&last_time) = self.frame_times.back() {
            let interval_us = time.duration_since(last_time).as_micros() as u64;
            self.min_interval_us = self.min_interval_us.min(interval_us);
            self.max_interval_us = self.max_interval_us.max(interval_us);

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

    fn average_fps(&self) -> f64 {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        if elapsed > 0.0 { self.total_frames as f64 / elapsed } else { 0.0 }
    }

    fn jitter_ms(&self) -> f64 {
        if self.intervals_us.len() < 2 {
            return 0.0;
        }
        let mean = self.intervals_us.iter().sum::<u64>() as f64 / self.intervals_us.len() as f64;
        let variance: f64 = self.intervals_us.iter()
            .map(|&x| { let d = x as f64 - mean; d * d })
            .sum::<f64>() / self.intervals_us.len() as f64;
        variance.sqrt() / 1000.0
    }

    fn report(&self) -> String {
        format!(
            "FPS: {:.1} (avg: {:.1}) | range: {:.1}-{:.1} | jitter: {:.2}ms | frames: {}",
            self.current_fps(),
            self.average_fps(),
            if self.max_interval_us > 0 { 1_000_000.0 / self.max_interval_us as f64 } else { 0.0 },
            if self.min_interval_us > 0 && self.min_interval_us < u64::MAX { 1_000_000.0 / self.min_interval_us as f64 } else { 0.0 },
            self.jitter_ms(),
            self.total_frames,
        )
    }
}

/// Рисование текста на буфере
fn draw_text(buffer: &mut [u8], width: usize, height: usize, stride: usize, text: &str, x: usize, y: usize) {
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
        ("A", [0b01110, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001, 0b10001]),
        ("v", [0b00000, 0b00000, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100]),
        ("g", [0b00000, 0b00000, 0b01111, 0b10001, 0b01111, 0b00001, 0b01110]),
        ("a", [0b00000, 0b00000, 0b01110, 0b00001, 0b01111, 0b10001, 0b01111]),
        ("r", [0b00000, 0b00000, 0b10110, 0b11001, 0b10000, 0b10000, 0b10000]),
        ("e", [0b00000, 0b00000, 0b01110, 0b10001, 0b11111, 0b10000, 0b01110]),
        ("n", [0b00000, 0b00000, 0b10110, 0b11001, 0b10001, 0b10001, 0b10001]),
        ("m", [0b00000, 0b00000, 0b11010, 0b10101, 0b10101, 0b10001, 0b10001]),
        ("s", [0b00000, 0b00000, 0b01110, 0b10000, 0b01110, 0b00001, 0b11110]),
        ("f", [0b00110, 0b01001, 0b01000, 0b11100, 0b01000, 0b01000, 0b01000]),
        ("i", [0b00100, 0b00000, 0b01100, 0b00100, 0b00100, 0b00100, 0b01110]),
        ("t", [0b01000, 0b01000, 0b11100, 0b01000, 0b01000, 0b01001, 0b00110]),
        ("j", [0b00010, 0b00000, 0b00110, 0b00010, 0b00010, 0b10010, 0b01100]),
        ("x", [0b00000, 0b00000, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001]),
        ("|", [0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100]),
        ("(", [0b00010, 0b00100, 0b01000, 0b01000, 0b01000, 0b00100, 0b00010]),
        (")", [0b01000, 0b00100, 0b00010, 0b00010, 0b00010, 0b00100, 0b01000]),
    ];

    let scale = 2usize;
    let mut cursor_x = x;

    for ch in text.chars() {
        let glyph = FONT.iter()
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
                                    // Для NV12: только Y плоскость (яркость)
                                    let idx = py * stride + px;
                                    if idx < buffer.len() {
                                        buffer[idx] = 255; // Белый текст
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

fn create_pipeline(device: &str) -> Result<(gst::Pipeline, Arc<Mutex<FpsAnalyzer>>)> {
    gst::init()?;

    let pipeline = gst::Pipeline::new();

    // Источник с ЯВНЫМ указанием формата, разрешения и FPS
    let src = gst::ElementFactory::make("v4l2src")
        .name("source")
        .property("device", device)
        .property("do-timestamp", true)
        .build()?;

    // Capsfilter для ВХОДА камеры - ЭТО КЛЮЧЕВОЙ МОМЕНТ!
    // Указываем NV12, 1920x1080, 60fps - как в вашей команде
    let capsfilter_src = gst::ElementFactory::make("capsfilter")
        .name("src_caps")
        .build()?;
    
    let caps_src = gst::Caps::builder("video/x-raw")
        .field("format", "NV12")
        .field("width", 1920i32)
        .field("height", 1080i32)
        .field("framerate", gst::Fraction::new(60, 1))
        .build();
    capsfilter_src.set_property("caps", &caps_src);

    // Identity для перехвата кадров
    let identity = gst::ElementFactory::make("identity")
        .name("identity")
        .build()?;

    // Вывод на экран (аналог fpsdisplaysink, но без встроенного FPS)
    let sink = gst::ElementFactory::make("autovideosink")
        .name("sink")
        .property("sync", false)  // Без синхронизации - максимальная скорость
        .build()?;

    pipeline.add_many([&src, &capsfilter_src, &identity, &sink])?;
    gst::Element::link_many([&src, &capsfilter_src, &identity, &sink])?;

    // Анализатор FPS
    let fps_analyzer = Arc::new(Mutex::new(FpsAnalyzer::new(120)));
    let fps_analyzer_clone = fps_analyzer.clone();

    let frame_counter = Arc::new(AtomicU64::new(0));
    let video_info: Arc<Mutex<Option<gst_video::VideoInfo>>> = Arc::new(Mutex::new(None));
    let video_info_clone = video_info.clone();

    // Probe для обработки кадров
    let identity_src_pad = identity.static_pad("src").unwrap();

    identity_src_pad.add_probe(gst::PadProbeType::BUFFER, move |pad, probe_info| {
        let capture_time = Instant::now();
        let frame_num = frame_counter.fetch_add(1, Ordering::SeqCst);

        // Обновляем FPS
        if let Ok(mut analyzer) = fps_analyzer_clone.lock() {
            analyzer.add_frame(capture_time);

            if frame_num > 0 && frame_num % 60 == 0 {
                println!("{}", analyzer.report());
            }
        }

        // Обрабатываем буфер
        if let Some(gst::PadProbeData::Buffer(ref mut buffer)) = probe_info.data {
            // Получаем video info при первом кадре
            let mut vi_guard = video_info_clone.lock().unwrap();
            if vi_guard.is_none() {
                if let Some(caps) = pad.current_caps() {
                    if let Ok(info) = gst_video::VideoInfo::from_caps(&caps) {
                        println!("===========================================");
                        println!("Video: {}x{}", info.width(), info.height());
                        println!("Format: {:?}", info.format());
                        println!("FPS: {:?}", info.fps());
                        println!("===========================================");
                        *vi_guard = Some(info);
                    }
                }
            }

            // Рисуем текст на кадре
            if let Some(ref info) = *vi_guard {
                let buffer_mut = buffer.make_mut();
                if let Ok(mut map) = buffer_mut.map_writable() {
                    let data = map.as_mut_slice();
                    let width = info.width() as usize;
                    let height = info.height() as usize;
                    
                    // Для NV12: stride может отличаться от width
                    let stride = info.stride()[0] as usize;

                    if let Ok(analyzer) = fps_analyzer_clone.lock() {
                        let text = format!(
                            "FPS: {:.1} (avg: {:.1}) frame: {}",
                            analyzer.current_fps(),
                            analyzer.average_fps(),
                            frame_num
                        );
                        draw_text(data, width, height, stride, &text, 20, 20);
                    }
                }
                
            }
        }

        gst::PadProbeReturn::Ok
    });

    Ok((pipeline, fps_analyzer))
}

/// Альтернативный pipeline с конвертацией в RGB для обработки через ndarray
fn create_pipeline_with_rgb_processing(device: &str) -> Result<(gst::Pipeline, Arc<Mutex<FpsAnalyzer>>)> {
    gst::init()?;

    let pipeline = gst::Pipeline::new();

    // Источник с указанием формата камеры
    let src = gst::ElementFactory::make("v4l2src")
        .property("device", device)
        .property("do-timestamp", true)
        .build()?;

    // Входной формат камеры
    let caps_src = gst::ElementFactory::make("capsfilter").build()?;
    caps_src.set_property("caps", &gst::Caps::builder("video/x-raw")
        .field("format", "NV12")
        .field("width", 1920i32)
        .field("height", 1080i32)
        .field("framerate", gst::Fraction::new(60, 1))
        .build());

    // Конвертер в RGB для обработки
    let convert = gst::ElementFactory::make("videoconvert").build()?;

    // RGB формат
    let caps_rgb = gst::ElementFactory::make("capsfilter").build()?;
    caps_rgb.set_property("caps", &gst::Caps::builder("video/x-raw")
        .field("format", "RGB")
        .build());

    let identity = gst::ElementFactory::make("identity").build()?;

    // Конвертер обратно для вывода
    let convert2 = gst::ElementFactory::make("videoconvert").build()?;

    let sink = gst::ElementFactory::make("autovideosink")
        .property("sync", false)
        .build()?;

    pipeline.add_many([&src, &caps_src, &convert, &caps_rgb, &identity, &convert2, &sink])?;
    gst::Element::link_many([&src, &caps_src, &convert, &caps_rgb, &identity, &convert2, &sink])?;

    let fps_analyzer = Arc::new(Mutex::new(FpsAnalyzer::new(120)));
    let fps_analyzer_clone = fps_analyzer.clone();
    let frame_counter = Arc::new(AtomicU64::new(0));
    let video_info: Arc<Mutex<Option<gst_video::VideoInfo>>> = Arc::new(Mutex::new(None));
    let video_info_clone = video_info.clone();

    let identity_src_pad = identity.static_pad("src").unwrap();

    identity_src_pad.add_probe(gst::PadProbeType::BUFFER, move |pad, probe_info| {
        let capture_time = Instant::now();
        let frame_num = frame_counter.fetch_add(1, Ordering::SeqCst);

        if let Ok(mut analyzer) = fps_analyzer_clone.lock() {
            analyzer.add_frame(capture_time);
            if frame_num > 0 && frame_num % 60 == 0 {
                println!("{}", analyzer.report());
            }
        }

        if let Some(gst::PadProbeData::Buffer(ref mut buffer)) = probe_info.data {
            let mut vi_guard = video_info_clone.lock().unwrap();
            if vi_guard.is_none() {
                if let Some(caps) = pad.current_caps() {
                    if let Ok(info) = gst_video::VideoInfo::from_caps(&caps) {
                        println!("Video: {}x{} {:?} @ {:?}", 
                            info.width(), info.height(), info.format(), info.fps());
                        *vi_guard = Some(info);
                    }
                }
            }

            // Обработка RGB через ndarray
            if let Some(ref info) = *vi_guard {
                let buffer_mut = buffer.make_mut();
                if let Ok(mut map) = buffer_mut.map_writable() {
                    let data = map.as_mut_slice();
                    let width = info.width() as usize;
                    let height = info.height() as usize;

                    // Создаём ArrayView3 для обработки
                    if let Ok(mut frame) = ndarray::ArrayViewMut3::from_shape(
                        (height, width, 3),
                        data
                    ) {
                        // Пример обработки: инвертируем верхний левый угол
                        let corner_size = 100.min(height).min(width);
                        for y in 0..corner_size {
                            for x in 0..corner_size {
                                for c in 0..3 {
                                    frame[[y, x, c]] = 255 - frame[[y, x, c]];
                                }
                            }
                        }

                        // Рисуем текст (в RGB)
                        if let Ok(analyzer) = fps_analyzer_clone.lock() {
                            let text = format!("FPS: {:.1}", analyzer.current_fps());
                            draw_text_rgb(data, width, height, &text, 20, 20);
                        }
                    }
                }
            }
        }

        gst::PadProbeReturn::Ok
    });

    Ok((pipeline, fps_analyzer))
}

fn draw_text_rgb(buffer: &mut [u8], width: usize, height: usize, text: &str, x: usize, y: usize) {
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
        (" ", [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000]),
        ("F", [0b11111, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000, 0b10000]),
        ("P", [0b11110, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000, 0b10000]),
        ("S", [0b01110, 0b10001, 0b10000, 0b01110, 0b00001, 0b10001, 0b01110]),
    ];

    let scale = 2usize;
    let mut cursor_x = x;

    for ch in text.chars() {
        let glyph = FONT.iter()
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
                                    let idx = (py * width + px) * 3;
                                    if idx + 2 < buffer.len() {
                                        buffer[idx] = 0;       // R
                                        buffer[idx + 1] = 255; // G
                                        buffer[idx + 2] = 0;   // B
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

fn main() -> Result<()> {
    println!("==========================================");
    println!("  GStreamer 60 FPS Video Capture");
    println!("==========================================\n");

    let device = "/dev/video11";

    if !std::path::Path::new(device).exists() {
        return Err(anyhow!("Камера {} не найдена!", device));
    }

    // Выберите один из вариантов:
    
    // 1. Быстрый вариант без конвертации (NV12 -> display)
    let (pipeline, fps_analyzer) = create_pipeline(device)?;
    
    // 2. С конвертацией в RGB для полноценной обработки через ndarray
    // let (pipeline, fps_analyzer) = create_pipeline_with_rgb_processing(device)?;

    println!("Запуск pipeline...");
    println!("Нажмите Ctrl+C для выхода\n");

    pipeline.set_state(gst::State::Playing)?;

    // Ctrl+C handler
    let running = Arc::new(AtomicBool::new(true));
    let running_clone = running.clone();
    
    std::thread::spawn(move || {
        let mut signals = signal_hook::iterator::Signals::new(&[signal_hook::consts::SIGINT])
            .expect("Ошибка регистрации сигналов");
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

    println!("\n========== Final Statistics ==========");
    if let Ok(analyzer) = fps_analyzer.lock() {
        println!("{}", analyzer.report());
    }
    println!("=======================================");

    pipeline.set_state(gst::State::Null)?;

    Ok(())
}