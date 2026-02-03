use std::{collections::VecDeque, time::Instant};

pub struct FpsAnalyzer {
    /// Время получения каждого кадра
    frame_times: VecDeque<Instant>,
    /// Интервалы между кадрами (в микросекундах)
    intervals_us: VecDeque<u64>,
    /// Максимальное количество сэмплов для анализа
    max_samples: usize,
    /// Общее количество кадров
    total_frames: u64,
    /// Время старта
    start_time: Instant,
    /// Минимальный интервал
    min_interval_us: u64,
    /// Максимальный интервал
    max_interval_us: u64,
    /// Сумма интервалов для среднего
    sum_intervals_us: u64,
}

impl FpsAnalyzer {
    pub fn new(max_samples: usize) -> Self {
        Self {
            frame_times: VecDeque::with_capacity(max_samples),
            intervals_us: VecDeque::with_capacity(max_samples),
            max_samples,
            total_frames: 0,
            start_time: Instant::now(),
            min_interval_us: u64::MAX,
            max_interval_us: 0,
            sum_intervals_us: 0,
        }
    }

    pub fn add_frame(&mut self, time: Instant) {
        // Вычисляем интервал
        if let Some(&last_time) = self.frame_times.back() {
            let interval = time.duration_since(last_time);
            let interval_us = interval.as_micros() as u64;

            // Обновляем статистику
            self.min_interval_us = self.min_interval_us.min(interval_us);
            self.max_interval_us = self.max_interval_us.max(interval_us);
            self.sum_intervals_us += interval_us;

            // Добавляем в очередь
            if self.intervals_us.len() >= self.max_samples {
                self.intervals_us.pop_front();
            }
            self.intervals_us.push_back(interval_us);
        }

        // Добавляем время кадра
        if self.frame_times.len() >= self.max_samples {
            self.frame_times.pop_front();
        }
        self.frame_times.push_back(time);

        self.total_frames += 1;
    }

    /// FPS за последние N кадров
    pub fn current_fps(&self) -> f64 {
        if self.intervals_us.is_empty() {
            return 0.0;
        }

        let avg_interval_us: f64 =
            self.intervals_us.iter().sum::<u64>() as f64 / self.intervals_us.len() as f64;

        if avg_interval_us > 0.0 {
            1_000_000.0 / avg_interval_us
        } else {
            0.0
        }
    }

    /// Средний FPS за всё время
    pub fn average_fps(&self) -> f64 {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            self.total_frames as f64 / elapsed
        } else {
            0.0
        }
    }

    /// Мгновенный FPS (между последними двумя кадрами)
    pub fn instant_fps(&self) -> f64 {
        if let Some(&last_interval) = self.intervals_us.back() {
            if last_interval > 0 {
                return 1_000_000.0 / last_interval as f64;
            }
        }
        0.0
    }

    /// Минимальный FPS
    pub fn min_fps(&self) -> f64 {
        if self.max_interval_us > 0 {
            1_000_000.0 / self.max_interval_us as f64
        } else {
            0.0
        }
    }

    /// Максимальный FPS
    pub fn max_fps(&self) -> f64 {
        if self.min_interval_us < u64::MAX && self.min_interval_us > 0 {
            1_000_000.0 / self.min_interval_us as f64
        } else {
            0.0
        }
    }

    /// Стандартное отклонение интервалов
    fn interval_std_dev(&self) -> f64 {
        if self.intervals_us.len() < 2 {
            return 0.0;
        }

        let mean = self.intervals_us.iter().sum::<u64>() as f64 / self.intervals_us.len() as f64;
        let variance: f64 = self
            .intervals_us
            .iter()
            .map(|&x| {
                let diff = x as f64 - mean;
                diff * diff
            })
            .sum::<f64>()
            / self.intervals_us.len() as f64;

        variance.sqrt()
    }

    /// Джиттер (вариация интервалов)
    pub fn jitter_ms(&self) -> f64 {
        self.interval_std_dev() / 1000.0
    }

    /// Полный отчёт
    pub fn report(&self) -> String {
        format!(
            "FPS: текущий={:.2}, средний={:.2}, мгновенный={:.2}\n\
             Диапазон FPS: {:.2} - {:.2}\n\
             Интервал между кадрами: мин={:.2}мс, макс={:.2}мс\n\
             Джиттер: {:.2}мс\n\
             Всего кадров: {}, время работы: {:.1}с",
            self.current_fps(),
            self.average_fps(),
            self.instant_fps(),
            self.min_fps(),
            self.max_fps(),
            self.min_interval_us as f64 / 1000.0,
            self.max_interval_us as f64 / 1000.0,
            self.jitter_ms(),
            self.total_frames,
            self.start_time.elapsed().as_secs_f64()
        )
    }
}
