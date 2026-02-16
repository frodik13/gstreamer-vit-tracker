mod nv12_convert;
mod selection_state;
mod user_commands;
mod raw_mode_guard;
mod timing_stats;
mod app_state;
mod tracker_context;
mod drawing;
mod pipeline;
mod pipeline_ir;
mod drawing_rgb;

use anyhow::{anyhow, Result};
use gstreamer as gst;
use gstreamer::prelude::*;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc};

use crate::pipeline_ir::create_pipeline_ir;
use crate::raw_mode_guard::*;

use pipeline::*;

const MODEL_PATH: &str = "/home/radxa/repos/rust/vit_tracker/models/object_tracking_vittrack_2023sep.rknn";

fn main() -> Result<()> {
    println!("==========================================\r");
    println!("   VitTrack - Interactive Selection\r");
    println!("==========================================\r\n");

    let device = "/dev/video21";

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
    let (pipeline, _ctx, _stats) = create_pipeline_ir(device, rx)?;

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