use gstreamer as gst;
use gstreamer::prelude::*;
use anyhow::{Result};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use crate::drawing::*;
use crate::nv12_convert::*;
use crate::{timing_stats::TimingStats, tracker_context::TrackerContext, user_commands::UserCommand};

const MODEL_PATH: &str = "/home/radxa/repos/rust/vit_tracker/models/object_tracking_vittrack_2023sep.rknn";

pub fn create_pipeline(
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

    let queue = gst::ElementFactory::make("queue")
        .property("max-size-buffers", 2u32) // Буфер на 2 кадра
        .property_from_str("leaky", "2") // downstream (если не успеваем отображать - дропаем старые кадры вывода, но не тормозим обработку)
        .build()?;

    let sink = gst::ElementFactory::make("kmssink")
        .property("sync", false)
        .property("connector-id", 231i32)
        .property("plane-id", 72i32)
        .build()?;

    pipeline.add_many([&src, &caps, &identity, &queue, &sink])?;
    gst::Element::link_many([&src, &caps, &identity, &queue, &sink])?;

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
        let (w, h) = (width as usize, heigth as usize);

        // Конвертация NV12 -> BGR (полный кадр, но параллельно)
        let t0 = Instant::now();
        let rgb = nv12_full_to_rgb_parallel(data, w, h);
        let conv_time = t0.elapsed().as_micros() as u64;

        // Трекинг
        let t1 = Instant::now();
        let (bbox, state_name, score, selection) = {
            let mut ctx = ctx_clone.lock().unwrap();
            let result = ctx.process_frame(&rgb.view());
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
        draw_background_nv12(data, w, h, 10, 10, 400, 80, 150);
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