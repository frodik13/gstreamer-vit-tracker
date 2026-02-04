use ndarray::ArrayView3;
use vit_tracker::{BBox, VitTrack};
use anyhow::{anyhow, Result};

use crate::{app_state::AppState, selection_state::{SelectionPhase, SelectionState}, user_commands::UserCommand};

pub struct TrackerContext {
    pub(crate) tracker: VitTrack,
    pub(crate) state: AppState,
    pub(crate) selection: SelectionState,
    pub(crate) current_bbox: Option<BBox>,
    pub(crate) current_score: f32,
    pub(crate) frame_width: i32,
    pub(crate) frame_height: i32,
    pub(crate) pending_confirm: bool,
}

impl TrackerContext {
    pub fn new(model_path: &str, width: i32, height: i32) -> Result<Self> {
        println!("Loading model: {}\r", model_path);
        let tracker = VitTrack::new(model_path).map_err(|e| anyhow!("Failed: {:?}", e))?;
        println!("Model loaded successfully\r");

        Ok(Self {
            tracker,
            state: AppState::Selecting,
            selection: SelectionState::new(width, height),
            current_bbox: None,
            current_score: 0.0,
            frame_width: width,
            frame_height: height,
            pending_confirm: false,
        })
    }

    pub fn handle_command(&mut self, cmd: UserCommand) {
        match cmd {
            UserCommand::MoveUp(fast) => {
                self.selection.move_cursor(0, -1, fast, self.frame_width, self.frame_height);
            }
            UserCommand::MoveDown(fast) => {
                self.selection.move_cursor(0, 1, fast, self.frame_width, self.frame_height);
            }
            UserCommand::MoveLeft(fast) => {
                self.selection.move_cursor(-1, 0, fast, self.frame_width, self.frame_height);
            }
            UserCommand::MoveRight(fast) => {
                self.selection.move_cursor(1, 0, fast, self.frame_width, self.frame_height);
            }
            UserCommand::Confirm => {
                self.pending_confirm = true;
            }
            UserCommand::Cancel => {
                self.state = AppState::Selecting;
                self.selection = SelectionState::new(self.frame_width, self.frame_height);
                self.current_bbox = None;
                println!("\rReset to selection mode\r");
            }
            UserCommand::Quit => {}
        }
    }

    /// Обрабатывает кадр - принимает ПОЛНЫЙ кадр в BGR
    pub fn process_frame(&mut self, full_image: &ArrayView3<u8>) -> Option<BBox> {
        match self.state {
            AppState::Selecting => {
                if self.pending_confirm {
                    self.pending_confirm = false;

                    match self.selection.phase {
                        SelectionPhase::MovingToStart => {
                            self.selection.start_x = self.selection.cursor_x;
                            self.selection.start_y = self.selection.cursor_y;
                            self.selection.phase = SelectionPhase::SelectingArea;
                            println!(
                                "\r*** Start point set at ({}, {}) ***\r",
                                self.selection.start_x, self.selection.start_y
                            );
                            println!("\rNow move to the SECOND corner and press Enter\r");
                        }
                        SelectionPhase::SelectingArea => {
                            let bbox = self.selection.get_bbox();
                            println!(
                                "\r*** Initializing tracker with bbox: x={}, y={}, w={}, h={} ***\r",
                                bbox.x, bbox.y, bbox.width, bbox.height
                            );

                            self.tracker.init(full_image, bbox);

                            match self.tracker.update(full_image) {
                                Ok(result) => {
                                    println!("\rInit result: score={:.3}\r", result.score);
                                    if result.success && result.score > 0.25 {
                                        self.current_bbox = Some(BBox::from_array(&result.bbox));
                                        self.current_score = result.score;
                                        self.state = AppState::Tracking;
                                        println!("\r*** TRACKING STARTED! ***\r");
                                        return self.current_bbox;
                                    } else {
                                        println!("\rLow score - please try selecting a different area\r");
                                        self.selection =
                                            SelectionState::new(self.frame_width, self.frame_height);
                                    }
                                }
                                Err(e) => {
                                    println!("\rTracker error: {:?}\r", e);
                                    self.selection =
                                        SelectionState::new(self.frame_width, self.frame_height);
                                }
                            }
                        }
                    }
                }
                None
            }

            AppState::Tracking => {
                self.pending_confirm = false;

                match self.tracker.update(full_image) {
                    Ok(result) => {
                        if result.success && result.score > 0.25 {
                            let bbox = BBox::from_array(&result.bbox);
                            self.current_bbox = Some(bbox);
                            self.current_score = result.score;
                            Some(bbox)
                        } else {
                            println!("\rTrack lost (score={:.2})\r", result.score);
                            self.state = AppState::Lost { frames: 0 };
                            self.current_score = 0.0;
                            None
                        }
                    }
                    Err(_) => {
                        println!("\rTracker error\r");
                        self.state = AppState::Lost { frames: 0 };
                        None
                    }
                }
            }

            AppState::Lost { frames } => {
                self.pending_confirm = false;
                if frames > 60 {
                    println!("\rAuto-reset to selection mode\r");
                    self.state = AppState::Selecting;
                    self.selection = SelectionState::new(self.frame_width, self.frame_height);
                    self.current_bbox = None;
                } else {
                    self.state = AppState::Lost { frames: frames + 1 };
                }
                None
            }
        }
    }

    pub fn state_name(&self) -> &'static str {
        match self.state {
            AppState::Selecting => match self.selection.phase {
                SelectionPhase::MovingToStart => "SELECT START",
                SelectionPhase::SelectingArea => "SELECT END",
            },
            AppState::Tracking => "TRACKING",
            AppState::Lost { .. } => "LOST",
        }
    }
}