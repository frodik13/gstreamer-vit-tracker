use anyhow::{anyhow, Result};
use crossbeam_channel::{bounded, Receiver, Sender};
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use gstreamer_video as gst_video;
use ndarray::Array3;
use opencv::{
    core::{self, Mat},
    highgui,
    imgproc,
    prelude::*,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

/// Структура для хранения кадра как ndarray
struct Frame {
    data: Array3<u8>,
}

impl Frame {
    fn from_raw(raw_data: &[u8], width: usize, height: usize, channels: usize) -> Result<Self> {
        let expected_size = height * width * channels;
        if raw_data.len() != expected_size {
            return Err(anyhow!(
                "Неверный размер данных: ожидается {}, получено {}",
                expected_size,
                raw_data.len()
            ));
        }

        // Создаём Array3 с формой [height, width, channels]
        let data = Array3::from_shape_vec((height, width, channels), raw_data.to_vec())?;

        Ok(Self { data })
    }

    fn shape(&self) -> (usize, usize, usize) {
        let shape = self.data.shape();
        (shape[0], shape[1], shape[2])
    }
}

/// Инициализация GStreamer пайплайна
fn create_pipeline(frame_sender: Sender<Frame>, device: &str) -> Result<gst::Pipeline> {
    gst::init()?;

    // Создаём элементы вручную
    let pipeline = gst::Pipeline::new();

    // Источник видео
    let src = gst::ElementFactory::make("v4l2src")
        .name("source")
        .property("device", device)
        .property("do-timestamp", true)
        .build()?;

    // Декодер (для MJPEG и других форматов)
    let decodebin = gst::ElementFactory::make("decodebin")
        .name("decoder")
        .build()?;

    // Конвертер видео
    let convert = gst::ElementFactory::make("videoconvert")
        .name("convert")
        .build()?;

    // Масштабирование
    let scale = gst::ElementFactory::make("videoscale")
        .name("scale")
        .build()?;

    // Capsfilter для указания выходного формата
    let capsfilter = gst::ElementFactory::make("capsfilter")
        .name("filter")
        .build()?;

    let caps = gst::Caps::builder("video/x-raw")
        .field("format", "RGB")
        .build();
    capsfilter.set_property("caps", &caps);

    // AppSink
    let appsink = gst::ElementFactory::make("appsink")
        .name("sink")
        .property("emit-signals", true)
        .property("sync", false)
        .property("max-buffers", 2u32)
        .property("drop", true)
        .build()?;

    // Добавляем элементы в pipeline
    pipeline.add_many([&src, &decodebin, &convert, &scale, &capsfilter, &appsink])?;

    // Связываем src -> decodebin
    src.link(&decodebin)?;

    // decodebin имеет динамические pads, нужен callback
    let convert_weak = convert.downgrade();
    decodebin.connect_pad_added(move |_decodebin, src_pad| {
        let Some(convert) = convert_weak.upgrade() else {
            return;
        };

        let sink_pad = convert
            .static_pad("sink")
            .expect("convert не имеет sink pad");

        if sink_pad.is_linked() {
            return;
        }

        let caps = src_pad.current_caps();
        if let Some(caps) = caps {
            let structure = caps.structure(0);
            if let Some(structure) = structure {
                let name = structure.name();
                if name.starts_with("video/") {
                    if let Err(e) = src_pad.link(&sink_pad) {
                        eprintln!("Не удалось связать pads: {:?}", e);
                    } else {
                        println!("Связаны pads: video");
                    }
                }
            }
        }
    });

    // Связываем остальные элементы
    convert.link(&scale)?;
    scale.link(&capsfilter)?;
    capsfilter.link(&appsink)?;

    // Настраиваем appsink
    let appsink = appsink
        .downcast::<gst_app::AppSink>()
        .map_err(|_| anyhow!("Не удалось преобразовать в AppSink"))?;

    let sender = frame_sender;
    appsink.set_callbacks(
        gst_app::AppSinkCallbacks::builder()
            .new_sample(move |sink| {
                if let Err(e) = process_sample(sink, &sender) {
                    eprintln!("Ошибка обработки кадра: {}", e);
                }
                Ok(gst::FlowSuccess::Ok)
            })
            .build(),
    );

    Ok(pipeline)
}

/// Альтернативный pipeline без decodebin (для raw форматов)
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
                if let Err(e) = process_sample(sink, &sender) {
                    eprintln!("Ошибка: {}", e);
                }
                Ok(gst::FlowSuccess::Ok)
            })
            .build(),
    );

    Ok(pipeline)
}

/// Pipeline для MJPEG камер
fn create_mjpeg_pipeline(frame_sender: Sender<Frame>, device: &str) -> Result<gst::Pipeline> {
    gst::init()?;

    let pipeline = gst::Pipeline::new();

    let src = gst::ElementFactory::make("v4l2src")
        .name("source")
        .property("device", device)
        .build()?;

    let jpegdec = gst::ElementFactory::make("jpegdec")
        .name("decoder")
        .build()?;

    let convert = gst::ElementFactory::make("videoconvert")
        .name("convert")
        .build()?;

    let capsfilter = gst::ElementFactory::make("capsfilter")
        .name("filter")
        .build()?;

    let caps = gst::Caps::builder("video/x-raw")
        .field("format", "RGB")
        .build();
    capsfilter.set_property("caps", &caps);

    let appsink = gst::ElementFactory::make("appsink")
        .name("sink")
        .property("emit-signals", true)
        .property("sync", false)
        .property("max-buffers", 2u32)
        .property("drop", true)
        .build()?;

    pipeline.add_many([&src, &jpegdec, &convert, &capsfilter, &appsink])?;

    gst::Element::link_many([&src, &jpegdec, &convert, &capsfilter, &appsink])?;

    let appsink = appsink
        .downcast::<gst_app::AppSink>()
        .map_err(|_| anyhow!("Не удалось преобразовать в AppSink"))?;

    let sender = frame_sender;
    appsink.set_callbacks(
        gst_app::AppSinkCallbacks::builder()
            .new_sample(move |sink| {
                if let Err(e) = process_sample(sink, &sender) {
                    eprintln!("Ошибка: {}", e);
                }
                Ok(gst::FlowSuccess::Ok)
            })
            .build(),
    );

    Ok(pipeline)
}

/// Обработка сэмпла
fn process_sample(sink: &gst_app::AppSink, frame_sender: &Sender<Frame>) -> Result<()> {
    let sample = sink
        .pull_sample()
        .map_err(|_| anyhow!("Не удалось получить sample"))?;

    let buffer = sample.buffer().ok_or_else(|| anyhow!("Нет буфера"))?;

    let caps = sample.caps().ok_or_else(|| anyhow!("Нет caps"))?;

    let video_info = gst_video::VideoInfo::from_caps(caps)
        .map_err(|_| anyhow!("Не удалось получить VideoInfo"))?;

    let width = video_info.width() as usize;
    let height = video_info.height() as usize;

    let map = buffer
        .map_readable()
        .map_err(|_| anyhow!("Не удалось замапить буфер"))?;

    let frame = Frame::from_raw(map.as_slice(), width, height, 3)?;

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

/// Цикл отображения
fn display_loop(frame_receiver: Receiver<Frame>, running: Arc<AtomicBool>) -> Result<()> {
    let window_name = "Camera";
    highgui::named_window(window_name, highgui::WINDOW_AUTOSIZE)?;

    println!("Нажмите 'q' или ESC для выхода");

    let mut frame_count = 0u64;
    let mut last_time = std::time::Instant::now();
    let mut average_fps = 0.0;
    let mut fps_vec = Vec::new();

    while running.load(Ordering::SeqCst) {
        match frame_receiver.recv_timeout(Duration::from_millis(50)) {
            Ok(frame) => {
                frame_count += 1;

                let now = std::time::Instant::now();
                

                let (height, width, _) = frame.shape();

                if frame_count % 30 == 0 {
                    println!("Кадр #{}: {}x{}, FPS: {:.1}", frame_count, width, height, average_fps);
                }

                let rgb_mat = array3_to_mat(&frame.data)?;

                let mut bgr_mat = Mat::default();
                imgproc::cvt_color(&rgb_mat, &mut bgr_mat, imgproc::COLOR_RGB2BGR, 0)?;

                let text = format!("Frame: {} FPS: {:.1}", frame_count, average_fps);
                imgproc::put_text(
                    &mut bgr_mat,
                    &text,
                    core::Point::new(10, 30),
                    imgproc::FONT_HERSHEY_SIMPLEX,
                    0.7,
                    core::Scalar::new(0.0, 255.0, 0.0, 0.0),
                    2,
                    imgproc::LINE_AA,
                    false,
                )?;

                highgui::imshow(window_name, &bgr_mat)?;
                let fps = 1.0 / now.duration_since(last_time).as_secs_f64();
                last_time = now;

                fps_vec.push(fps);
                if fps_vec.len() > 30 {
                    average_fps = fps_vec.iter().sum::<f64>() / fps_vec.len() as f64;
                    fps_vec.remove(0);
                }
            }
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                break;
            }
        }

        let key = highgui::wait_key(1)?;
        if key == 'q' as i32 || key == 27 {
            running.store(false, Ordering::SeqCst);
            break;
        }
    }

    highgui::destroy_all_windows()?;
    Ok(())
}

fn main() -> Result<()> {
    println!("===========================================");
    println!("Запуск захвата видео с камеры");
    println!("===========================================");

    let device = "/dev/video11";

    if !std::path::Path::new(device).exists() {
        eprintln!("Камера {} не найдена!", device);
        println!("Доступные устройства:");
        for entry in std::fs::read_dir("/dev")? {
            if let Ok(entry) = entry {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                if name_str.starts_with("video") {
                    println!("  /dev/{}", name_str);
                }
            }
        }
        return Err(anyhow!("Камера не найдена"));
    }

    let (frame_sender, frame_receiver) = bounded::<Frame>(4);

    let running = Arc::new(AtomicBool::new(true));
    let running_display = running.clone();
    let running_bus = running.clone();

    // Пробуем разные pipelines
    println!("Пробуем простой pipeline...");
    let pipeline = match create_simple_pipeline(frame_sender.clone(), device) {
        Ok(p) => p,
        Err(e) => {
            println!("Простой pipeline не работает: {}", e);
            println!("Пробуем MJPEG pipeline...");
            match create_mjpeg_pipeline(frame_sender.clone(), device) {
                Ok(p) => p,
                Err(e) => {
                    println!("MJPEG pipeline не работает: {}", e);
                    println!("Пробуем decodebin pipeline...");
                    create_pipeline(frame_sender, device)?
                }
            }
        }
    };

    // Запускаем отображение
    let display_handle = thread::spawn(move || {
        if let Err(e) = display_loop(frame_receiver, running_display) {
            eprintln!("Ошибка отображения: {}", e);
        }
    });

    thread::sleep(Duration::from_millis(300));

    // Запускаем pipeline
    println!("Запуск pipeline...");
    pipeline.set_state(gst::State::Playing)?;

    // Ждём перехода в Playing
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
                    if let Some(debug) = err.debug() {
                        eprintln!("Debug: {}", debug);
                    }
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