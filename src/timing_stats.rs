use std::collections::VecDeque;

pub struct TimingStats {
    intervals: VecDeque<u64>,
    conv_times: VecDeque<u64>,
    track_times: VecDeque<u64>,
}

impl TimingStats {
    pub fn new() -> Self {
        Self {
            intervals: VecDeque::with_capacity(120),
            conv_times: VecDeque::with_capacity(120),
            track_times: VecDeque::with_capacity(120),
        }
    }

    pub fn add_interval(&mut self, v: u64) {
        if self.intervals.len() >= 120 {
            self.intervals.pop_front();
        }
        self.intervals.push_back(v);
    }

    pub fn add_times(&mut self, conv: u64, track: u64) {
        if self.conv_times.len() >= 120 {
            self.conv_times.pop_front();
        }
        if self.track_times.len() >= 120 {
            self.track_times.pop_front();
        }
        self.conv_times.push_back(conv);
        self.track_times.push_back(track);
    }

    pub fn fps(&self) -> f64 {
        if self.intervals.is_empty() {
            return 0.0;
        }
        let avg = self.intervals.iter().sum::<u64>() as f64 / self.intervals.len() as f64;
        if avg > 0.0 {
            1_000_000.0 / avg
        } else {
            0.0
        }
    }

    pub fn avg_conv_ms(&self) -> f64 {
        if self.conv_times.is_empty() {
            return 0.0;
        }
        self.conv_times.iter().sum::<u64>() as f64 / self.conv_times.len() as f64 / 1000.0
    }

    pub fn avg_track_ms(&self) -> f64 {
        if self.track_times.is_empty() {
            return 0.0;
        }
        self.track_times.iter().sum::<u64>() as f64 / self.track_times.len() as f64 / 1000.0
    }
}