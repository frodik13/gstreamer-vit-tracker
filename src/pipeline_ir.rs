// pipeline_ir.rs
use gstreamer as gst;
use gstreamer::prelude::*;
use anyhow::Result;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use crate::drawing_rgb::*;
use crate::{timing_stats::TimingStats, tracker_context::TrackerContext, user_commands::UserCommand};

const MODEL_PATH: &str = "/home/radxa/repos/rust/vit_tracker/models/object_tracking_vittrack_2023sep.rknn";

pub fn create_pipeline_ir(
    device: &str,
    cmd_rx: std::sync::mpsc::Receiver<UserCommand>,
) -> Result<(gst::Pipeline, Arc<Mutex<TrackerContext>>, Arc<Mutex<TimingStats>>)> {
    gst::init()?;

    let pipeline = gst::Pipeline::new();

    let src = gst::ElementFactory::make("v4l2src")
        .property("device", device)
        .property("do-timestamp", true)
        .property_from_str("io-mode", "4") // dmabuf
        .build()?;

    let width = 640;
    let height = 512;
    let display_width = 1280;
    let display_height = 1024;

    let caps_src = gst::ElementFactory::make("capsfilter").build()?;
    caps_src.set_property(
        "caps",
        &gst::Caps::builder("video/x-raw")
            .field("format", "YUY2")
            .field("width", width)
            .field("height", height)
            .field("framerate", gst::Fraction::new(60, 1))
            .build(),
    );

    let convert = gst::ElementFactory::make("videoconvert")
        .property("n-threads", 4u32)
        .build()?;

    let caps_rgb = gst::ElementFactory::make("capsfilter").build()?;
    caps_rgb.set_property(
        "caps",
        &gst::Caps::builder("video/x-raw")
            .field("format", "RGB")
            .field("width", width)
            .field("height", height)
            .field("framerate", gst::Fraction::new(60, 1))
            .build(),
    );

    // identity с probe — работает с 640x512 RGB (трекинг + отрисовка)
    let identity = gst::ElementFactory::make("identity").build()?;

    // Масштабирование ПОСЛЕ обработки
    let scaler = gst::ElementFactory::make("rgaconvert").build()?;

    let caps_scaled = gst::ElementFactory::make("capsfilter").build()?;
    caps_scaled.set_property(
        "caps",
        &gst::Caps::builder("video/x-raw")
            .field("format", "RGB")
            .field("width", display_width)
            .field("height", display_height)
            .field("framerate", gst::Fraction::new(60, 1))
            .build(),
    );

    let queue = gst::ElementFactory::make("queue")
        .property("max-size-buffers", 3u32)
        .property_from_str("leaky", "2")
        .build()?;

    let sink = gst::ElementFactory::make("kmssink")
        .property("sync", false)
        .property("connector-id", 231i32)
        .property("plane-id", 72i32)
        .build()?;

    pipeline.add_many([&src, &caps_src, &convert, &caps_rgb, &identity, &scaler, &caps_scaled, &queue, &sink])?;
    gst::Element::link_many([&src, &caps_src, &convert, &caps_rgb, &identity, &scaler, &caps_scaled, &queue, &sink])?;

    let ctx = Arc::new(Mutex::new(TrackerContext::new(MODEL_PATH, width, height)?));
    let stats = Arc::new(Mutex::new(TimingStats::new()));

    let ctx_clone = ctx.clone();
    let stats_clone = stats.clone();
    let cmd_rx = Arc::new(Mutex::new(cmd_rx));
    let frame_num = Arc::new(AtomicU64::new(0));
    let last_time: Arc<Mutex<Option<Instant>>> = Arc::new(Mutex::new(None));

    let pad = identity.static_pad("src").unwrap();

    pad.add_probe(gst::PadProbeType::BUFFER, move |_pad, probe_info| {
        let probe_start = Instant::now();

        let now = Instant::now();
        {
            let mut last = last_time.lock().unwrap();
            if let Some(prev) = *last {
                stats_clone.lock().unwrap()
                    .add_interval(now.duration_since(prev).as_micros() as u64);
            }
            *last = Some(now);
        }

        let num = frame_num.fetch_add(1, Ordering::SeqCst);

        if let Ok(rx) = cmd_rx.lock() {
            while let Ok(cmd) = rx.try_recv() {
                ctx_clone.lock().unwrap().handle_command(cmd);
            }
        }

        let buffer = match &mut probe_info.data {
            Some(gst::PadProbeData::Buffer(b)) => b,
            _ => return gst::PadProbeReturn::Ok,
        };

        let t_map = Instant::now();

        let buffer = buffer.make_mut();
        let Ok(mut map) = buffer.map_writable() else {
            return gst::PadProbeReturn::Ok;
        };

        let map_time = t_map.elapsed().as_micros() as u64;

        let data = map.as_mut_slice();
        let (w, h) = (width as usize, height as usize);

        // Данные уже в RGB — не нужна конвертация!
        // Создаём ndarray view для трекера
        let t0 = Instant::now();
        let t_view = Instant::now();
        let rgb_view = ndarray::ArrayView3::from_shape((h, w, 3), data).unwrap();
        let view_time = t_view.elapsed().as_micros() as u64;
        let conv_time = t0.elapsed().as_micros() as u64;

        // Трекинг
        let t1 = Instant::now();
        let (bbox, state_name, score, selection) = {
            let mut ctx = ctx_clone.lock().unwrap();
            let result = ctx.process_frame(&rgb_view);
            (
                result,
                ctx.state_name().to_string(),
                ctx.current_score,
                ctx.selection.clone(),
            )
        };
        let track_time = t1.elapsed().as_micros() as u64;

        stats_clone.lock().unwrap().add_times(conv_time, track_time);

         let t_draw = Instant::now();
        // Отрисовка на RGB
        let t1 = Instant::now();
        // draw_background_rgb(data, w, h, 10, 10, 350, 80, 150);
        let bg_time = t1.elapsed().as_micros();
        let t2 = Instant::now();
        draw_text_rgb(data, w, h, &state_name, 15, 15, 2, 255);

        let (fps, conv_ms, track_ms) = {
            let s = stats_clone.lock().unwrap();
            (s.fps(), s.avg_conv_ms(), s.avg_track_ms())
        };

        draw_text_rgb(data, w, h, &format!("FPS: {:.0}", fps), 15, 40, 2, 255);
        draw_text_rgb(data, w, h,
            &format!("trk:{:.1}ms", track_ms),
            15, 65, 1, 200);
        
        let text_time = t2.elapsed().as_micros();

        if state_name == "TRACKING" {
            draw_text_rgb(data, w, h,
                &format!("score: {:.0}%", score * 100.0),
                200, 15, 2, 255);
        }

        if state_name.starts_with("SELECT") {
            draw_cursor_rgb(data, w, h, selection.cursor_x, selection.cursor_y);
            draw_selection_rgb(data, w, h, &selection);
        }

        let t3 = Instant::now();
        if let Some(b) = bbox {
            draw_rect_rgb(data, w, h, b.x, b.y, b.width, b.height, 3, 0, 255, 0);
            draw_crosshair_rgb(data, w, h, b.x + b.width/2, b.y + b.height/2, 15, 0, 255, 0);
        } else if state_name == "TRACKING" {
            if let Some(b) = ctx_clone.lock().unwrap().current_bbox {
                draw_rect_rgb(data, w, h, b.x, b.y, b.width, b.height, 3, 0, 255, 0);
                draw_crosshair_rgb(data, w, h, b.x + b.width/2, b.y + b.height/2, 15, 0, 255, 0);
            }
        }

        let bbox_time = t3.elapsed().as_micros();

        let draw_time = t_draw.elapsed().as_micros() as u64;

        let total_time = probe_start.elapsed().as_micros() as u64;

        if num % 60 == 0 && num > 0 {
            println!(
                "\r[{}] FPS: {:.0} | track: {:.1}ms | draw: {:.1}ms (bg:{:.1} txt:{:.1} bbox:{:.1})\r",
                state_name, fps,
                track_time as f64 / 1000.0,
                draw_time as f64 / 1000.0,
                bg_time as f64 / 1000.0,
                text_time as f64 / 1000.0,
                bbox_time as f64 / 1000.0,
            );
        }

        // if num % 120 == 0 && num > 0 {
        //     println!("\r[{}] FPS: {:.0} | track: {:.1}ms\r",
        //         state_name, fps, track_ms);
        // }

        gst::PadProbeReturn::Ok
    });

    Ok((pipeline, ctx, stats))
}