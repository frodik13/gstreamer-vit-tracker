use anyhow::{Result, anyhow};
use ndarray::Array3;
use std::time::Instant;

/// Статистика по времени кадров
#[derive(Debug, Clone)]
pub struct FrameStats {
    /// Время получения кадра от камеры
    pub(crate) capture_time: Instant,
    /// Timestamp из GStreamer (PTS)
    pub(crate) gst_pts: Option<u64>,
    /// Timestamp из GStreamer (DTS)
    pub(crate) gst_dts: Option<u64>,
    /// Размер буфера
    pub(crate) buffer_size: usize,
    /// Номер кадра
    pub(crate) frame_number: u64,
}

/// Структура для хранения кадра
pub struct Frame {
    pub(crate) data: Array3<u8>,
    pub(crate) stats: FrameStats,
}

impl Frame {
    pub fn from_raw(
        raw_data: &[u8],
        width: usize,
        height: usize,
        channels: usize,
        stats: FrameStats,
    ) -> Result<Self> {
        let expected_size = height * width * channels;
        if raw_data.len() != expected_size {
            return Err(anyhow!(
                "Неверный размер данных: ожидается {}, получено {}",
                expected_size,
                raw_data.len()
            ));
        }

        let data = Array3::from_shape_vec((height, width, channels), raw_data.to_vec())?;

        Ok(Self { data, stats })
    }

    pub fn shape(&self) -> (usize, usize, usize) {
        let shape = self.data.shape();
        (shape[0], shape[1], shape[2])
    }
}
