use anyhow::{Result, anyhow};
use crossbeam_channel::{Receiver, Sender, bounded};
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use gstreamer_video as gst_video;
use ndarray::Array3;
use opencv::{
    core::{self, Mat},
    highgui, imgproc,
};

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

mod fps_analyzer;
mod frame;

use crate::fps_analyzer::FpsAnalyzer;
use crate::frame::{Frame, FrameStats};

/// Глобальный счётчик кадров для callback'а
static FRAME_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Создание pipeline
fn create_simple_pipeline(frame_sender: Sender<Frame>, device: &str) -> Result<gst::Pipeline> {
    gst::init()?;

    let pipeline = gst::Pipeline::new();

    let src = gst::ElementFactory::make("v4l2src")
        .name("source")
        .property("device", device)
        .build()?;

    let convert = gst::ElementFactory::make("videoconvert")
        .name("convert")
        .build()?;

    let capsfilter = gst::ElementFactory::make("capsfilter")
        .name("filter")
        .build()?;

    let caps = gst::Caps::builder("video/x-raw")
        .field("format", "RGB")
        .field("width", 1920)
        .field("height", 1080)
        .field("framerate", &gstreamer::Fraction::new(60, 1))
        .build();
    capsfilter.set_property("caps", &caps);

    let appsink = gst::ElementFactory::make("appsink")
        .name("sink")
        .property("emit-signals", true)
        .property("sync", false)
        .property("max-buffers", 2u32)
        .property("drop", true)
        .build()?;

    pipeline.add_many([&src, &convert, &capsfilter, &appsink])?;
    gst::Element::link_many([&src, &convert, &capsfilter, &appsink])?;

    let appsink = appsink
        .downcast::<gst_app::AppSink>()
        .map_err(|_| anyhow!("Не удалось преобразовать в AppSink"))?;

    let sender = frame_sender;
    appsink.set_callbacks(
        gst_app::AppSinkCallbacks::builder()
            .new_sample(move |sink| {
                let capture_time = Instant::now();
                let frame_number = FRAME_COUNTER.fetch_add(1, Ordering::SeqCst);

                if let Err(e) = process_sample(sink, &sender, capture_time, frame_number) {
                    eprintln!("Ошибка: {}", e);
                }
                Ok(gst::FlowSuccess::Ok)
            })
            .build(),
    );

    Ok(pipeline)
}

/// Обработка сэмпла с замером времени
fn process_sample(
    sink: &gst_app::AppSink,
    frame_sender: &Sender<Frame>,
    capture_time: Instant,
    frame_number: u64,
) -> Result<()> {
    let sample = sink
        .pull_sample()
        .map_err(|_| anyhow!("Не удалось получить sample"))?;

    let buffer = sample.buffer().ok_or_else(|| anyhow!("Нет буфера"))?;

    // Получаем timestamps из GStreamer
    let gst_pts = buffer.pts().map(|pts| pts.nseconds());
    let gst_dts = buffer.dts().map(|dts| dts.nseconds());
    let buffer_size = buffer.size();

    let caps = sample.caps().ok_or_else(|| anyhow!("Нет caps"))?;

    let video_info = gst_video::VideoInfo::from_caps(caps)
        .map_err(|_| anyhow!("Не удалось получить VideoInfo"))?;

    let width = video_info.width() as usize;
    let height = video_info.height() as usize;

    let map = buffer
        .map_readable()
        .map_err(|_| anyhow!("Не удалось замапить буфер"))?;

    let stats = FrameStats {
        capture_time,
        gst_pts,
        gst_dts,
        buffer_size,
        frame_number,
    };

    let frame = Frame::from_raw(map.as_slice(), width, height, 3, stats)?;

    let _ = frame_sender.try_send(frame);

    Ok(())
}

/// Конвертация Array3 в Mat
fn array3_to_mat(arr: &Array3<u8>) -> Result<Mat> {
    let shape = arr.shape();
    let height = shape[0] as i32;
    let width = shape[1] as i32;

    let data = arr
        .as_slice()
        .ok_or_else(|| anyhow!("Массив не непрерывный"))?;

    let mat = unsafe {
        Mat::new_rows_cols_with_data_unsafe(
            height,
            width,
            core::CV_8UC3,
            data.as_ptr() as *mut std::ffi::c_void,
            (width * 3) as usize,
        )?
    };

    Ok(mat.clone())
}

/// Цикл отображения с замерами
fn display_loop(frame_receiver: Receiver<Frame>, running: Arc<AtomicBool>) -> Result<()> {
    let window_name = "Camera FPS Analysis";
    highgui::named_window(window_name, highgui::WINDOW_AUTOSIZE)?;

    println!("Нажмите 'q' или ESC для выхода");
    println!("Нажмите 'r' для сброса статистики");
    println!("Нажмите 'p' для вывода полного отчёта");
    println!();

    let mut fps_analyzer = FpsAnalyzer::new(100); // Анализ последних 100 кадров
    let mut display_update_time = Instant::now();

    while running.load(Ordering::SeqCst) {
        match frame_receiver.recv_timeout(Duration::from_millis(50)) {
            Ok(frame) => {
                // Добавляем время кадра в анализатор
                fps_analyzer.add_frame(frame.stats.capture_time);

                let (height, width, _) = frame.shape();

                // Выводим статистику каждые 2 секунды
                if display_update_time.elapsed() >= Duration::from_secs(2) {
                    println!("\n========== Статистика FPS ==========");
                    println!("{}", fps_analyzer.report());
                    println!("=====================================\n");
                    display_update_time = Instant::now();
                }

                // Отображение
                let rgb_mat = array3_to_mat(&frame.data)?;

                let mut bgr_mat = Mat::default();
                imgproc::cvt_color(&rgb_mat, &mut bgr_mat, imgproc::COLOR_RGB2BGR, 0)?;

                // Добавляем информацию на кадр
                let lines = [
                    format!("Frame: {}", frame.stats.frame_number),
                    format!("Current FPS: {:.2}", fps_analyzer.current_fps()),
                    format!("Average FPS: {:.2}", fps_analyzer.average_fps()),
                    format!("Instant FPS: {:.2}", fps_analyzer.instant_fps()),
                    format!(
                        "FPS Range: {:.1} - {:.1}",
                        fps_analyzer.min_fps(),
                        fps_analyzer.max_fps()
                    ),
                    format!("Jitter: {:.2} ms", fps_analyzer.jitter_ms()),
                    format!("Size: {}x{}", width, height),
                ];

                for (i, line) in lines.iter().enumerate() {
                    imgproc::put_text(
                        &mut bgr_mat,
                        line,
                        core::Point::new(10, 25 + i as i32 * 25),
                        imgproc::FONT_HERSHEY_SIMPLEX,
                        0.6,
                        core::Scalar::new(0.0, 255.0, 0.0, 0.0),
                        2,
                        imgproc::LINE_AA,
                        false,
                    )?;
                }

                // Показываем GStreamer timestamps если есть
                if let Some(pts) = frame.stats.gst_pts {
                    let pts_ms = pts as f64 / 1_000_000.0;
                    imgproc::put_text(
                        &mut bgr_mat,
                        &format!("PTS: {:.2} ms", pts_ms),
                        core::Point::new(10, height as i32 - 10),
                        imgproc::FONT_HERSHEY_SIMPLEX,
                        0.5,
                        core::Scalar::new(255.0, 255.0, 0.0, 0.0),
                        1,
                        imgproc::LINE_AA,
                        false,
                    )?;
                }

                highgui::imshow(window_name, &bgr_mat)?;
            }
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                break;
            }
        }

        let key = highgui::wait_key(1)?;
        match key {
            113 | 27 => {
                // 'q' или ESC
                println!("\n========== Финальная статистика ==========");
                println!("{}", fps_analyzer.report());
                println!("==========================================\n");
                running.store(false, Ordering::SeqCst);
                break;
            }
            114 => {
                // 'r' - сброс
                fps_analyzer = FpsAnalyzer::new(100);
                FRAME_COUNTER.store(0, Ordering::SeqCst);
                println!("Статистика сброшена");
            }
            112 => {
                // 'p' - печать отчёта
                println!("\n========== Текущая статистика ==========");
                println!("{}", fps_analyzer.report());
                println!("=========================================\n");
            }
            _ => {}
        }
    }

    highgui::destroy_all_windows()?;
    Ok(())
}

fn main() -> Result<()> {
    println!("===========================================");
    println!("   Анализ FPS камеры через GStreamer");
    println!("===========================================");
    println!();

    let device = "/dev/video11";

    if !std::path::Path::new(device).exists() {
        eprintln!("Камера {} не найдена!", device);
        return Err(anyhow!("Камера не найдена"));
    }

    let (frame_sender, frame_receiver) = bounded::<Frame>(4);

    let running = Arc::new(AtomicBool::new(true));
    let running_display = running.clone();
    let running_bus = running.clone();

    let pipeline = create_simple_pipeline(frame_sender, device)?;

    // Запускаем отображение
    let display_handle = std::thread::spawn(move || {
        if let Err(e) = display_loop(frame_receiver, running_display) {
            eprintln!("Ошибка отображения: {}", e);
        }
    });

    std::thread::sleep(Duration::from_millis(300));

    println!("Запуск pipeline...");
    pipeline.set_state(gst::State::Playing)?;

    let state_result = pipeline.state(gst::ClockTime::from_seconds(5));
    println!("Состояние: {:?}", state_result);

    // Обработка сообщений
    let bus = pipeline.bus().unwrap();

    while running_bus.load(Ordering::SeqCst) {
        if let Some(msg) = bus.timed_pop(gst::ClockTime::from_mseconds(100)) {
            use gst::MessageView;

            match msg.view() {
                MessageView::Eos(..) => {
                    println!("Конец потока");
                    running_bus.store(false, Ordering::SeqCst);
                    break;
                }
                MessageView::Error(err) => {
                    eprintln!("Ошибка GStreamer: {}", err.error());
                    running_bus.store(false, Ordering::SeqCst);
                    break;
                }
                MessageView::StateChanged(sc) => {
                    if sc.src().map(|s| s == &pipeline).unwrap_or(false) {
                        println!("Pipeline: {:?} -> {:?}", sc.old(), sc.current());
                    }
                }
                _ => {}
            }
        }
    }

    println!("Остановка...");
    pipeline.set_state(gst::State::Null)?;

    running.store(false, Ordering::SeqCst);
    let _ = display_handle.join();

    println!("Завершено");
    Ok(())
}
