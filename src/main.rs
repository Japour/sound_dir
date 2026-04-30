#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use eframe::egui::{self, Color32, Pos2, Rect, Vec2};
use global_hotkey::hotkey::{Code, HotKey, Modifiers};
use global_hotkey::{GlobalHotKeyEvent, GlobalHotKeyManager, HotKeyState};
use realfft::num_complex::Complex;
use realfft::{RealFftPlanner, RealToComplex};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tray_icon::menu::{Menu, MenuEvent, MenuItem};
use tray_icon::{TrayIcon, TrayIconBuilder};

// ───────── audio analysis ─────────

const FFT_SIZE: usize = 1024;
const HOP_SIZE: usize = 256; // меньше hop = чаще обновления (раз в ~5 мс)
const NUM_BINS: usize = FFT_SIZE / 2 + 1;

const BAND_HZ: [(f32, f32); 4] = [
    (20.0, 200.0),
    (200.0, 2000.0),
    (2000.0, 5000.0),
    (5000.0, 10000.0),
];

fn band_color(i: usize) -> Color32 {
    match i {
        0 => Color32::from_rgb(80, 170, 255),
        1 => Color32::from_rgb(80, 240, 130),
        2 => Color32::from_rgb(255, 110, 50),
        _ => Color32::from_rgb(255, 230, 90),
    }
}

const PCEN_S: f32 = 0.025;
const PCEN_ALPHA: f32 = 0.98;
const PCEN_DELTA: f32 = 2.0;
const PCEN_R: f32 = 0.5;
const PCEN_EPS: f32 = 1e-6;

struct BandLevels {
    energy: [AtomicU32; 4],
    direction: [AtomicU32; 4],
}

impl BandLevels {
    fn new() -> Self {
        Self {
            energy: std::array::from_fn(|_| AtomicU32::new(0)),
            direction: std::array::from_fn(|_| AtomicU32::new(0)),
        }
    }
    fn store_energy(&self, b: usize, v: f32) {
        self.energy[b].store(v.to_bits(), Ordering::Relaxed);
    }
    fn store_direction(&self, b: usize, v: f32) {
        self.direction[b].store(v.to_bits(), Ordering::Relaxed);
    }
    fn load_energy(&self, b: usize) -> f32 {
        f32::from_bits(self.energy[b].load(Ordering::Relaxed))
    }
    fn load_direction(&self, b: usize) -> f32 {
        f32::from_bits(self.direction[b].load(Ordering::Relaxed))
    }
}

struct ChannelState {
    ring: Vec<f32>,
    write_pos: usize,
    pcen_m: Vec<f32>,
    pcen_init: bool,
    fft_input: Vec<f32>,
    fft_output: Vec<Complex<f32>>,
    fft_scratch: Vec<Complex<f32>>,
}

impl ChannelState {
    fn new(fft: &dyn RealToComplex<f32>) -> Self {
        Self {
            ring: vec![0.0; FFT_SIZE],
            write_pos: 0,
            pcen_m: vec![0.0; NUM_BINS],
            pcen_init: false,
            fft_input: fft.make_input_vec(),
            fft_output: fft.make_output_vec(),
            fft_scratch: fft.make_scratch_vec(),
        }
    }
}

struct Analyzer {
    fft: Arc<dyn RealToComplex<f32>>,
    window: Vec<f32>,
    band_bins: [(usize, usize); 4],
    channels_in: usize,
    samples_until_hop: usize,
    per_channel: [ChannelState; 2],
    output: Arc<BandLevels>,
}

impl Analyzer {
    fn new(sample_rate: f32, channels_in: usize, output: Arc<BandLevels>) -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(FFT_SIZE);

        let window: Vec<f32> = (0..FFT_SIZE)
            .map(|n| {
                0.5 - 0.5 * (2.0 * std::f32::consts::PI * n as f32 / (FFT_SIZE - 1) as f32).cos()
            })
            .collect();

        let bin_hz = sample_rate / FFT_SIZE as f32;
        let band_bins: [(usize, usize); 4] = std::array::from_fn(|i| {
            let (lo, hi) = BAND_HZ[i];
            let lo_bin = ((lo / bin_hz) as usize).max(1);
            let hi_bin = ((hi / bin_hz).ceil() as usize).min(NUM_BINS - 1);
            (lo_bin, hi_bin)
        });

        let per_channel = [
            ChannelState::new(fft.as_ref()),
            ChannelState::new(fft.as_ref()),
        ];

        Self {
            fft,
            window,
            band_bins,
            channels_in,
            samples_until_hop: FFT_SIZE - 1,
            per_channel,
            output,
        }
    }

    fn process(&mut self, data: &[f32]) {
        let channels_in = self.channels_in;
        for frame in data.chunks(channels_in) {
            for ch in 0..2usize {
                let src = if ch < channels_in { ch } else { 0 };
                let s = frame[src];
                let cs = &mut self.per_channel[ch];
                cs.ring[cs.write_pos] = s;
                cs.write_pos = (cs.write_pos + 1) % FFT_SIZE;
            }
            if self.samples_until_hop == 0 {
                self.run_fft();
                self.samples_until_hop = HOP_SIZE - 1;
            } else {
                self.samples_until_hop -= 1;
            }
        }
    }

    fn run_fft(&mut self) {
        for ch in 0..2usize {
            let cs = &mut self.per_channel[ch];
            let start = cs.write_pos;
            for i in 0..FFT_SIZE {
                let idx = (start + i) % FFT_SIZE;
                cs.fft_input[i] = cs.ring[idx] * self.window[i];
            }
            if self
                .fft
                .process_with_scratch(&mut cs.fft_input, &mut cs.fft_output, &mut cs.fft_scratch)
                .is_err()
            {
                return;
            }
        }

        let (s0, s1) = self.per_channel.split_at_mut(1);
        let cs0 = &mut s0[0];
        let cs1 = &mut s1[0];
        let init0 = cs0.pcen_init;
        let init1 = cs1.pcen_init;

        for b in 0..4 {
            let (lo, hi) = self.band_bins[b];
            let mut sum_pcen = 0.0f32;
            let mut weighted_dir_num = 0.0f32;
            let mut weighted_dir_den = 0.0f32;

            for k in lo..hi {
                let l_mag = cs0.fft_output[k].norm();
                let r_mag = cs1.fft_output[k].norm();

                let m0 = if !init0 {
                    l_mag
                } else {
                    (1.0 - PCEN_S) * cs0.pcen_m[k] + PCEN_S * l_mag
                };
                cs0.pcen_m[k] = m0;
                let denom0 = (PCEN_EPS + m0).powf(PCEN_ALPHA);
                let l_pcen = (l_mag / denom0 + PCEN_DELTA).powf(PCEN_R) - PCEN_DELTA.powf(PCEN_R);

                let m1 = if !init1 {
                    r_mag
                } else {
                    (1.0 - PCEN_S) * cs1.pcen_m[k] + PCEN_S * r_mag
                };
                cs1.pcen_m[k] = m1;
                let denom1 = (PCEN_EPS + m1).powf(PCEN_ALPHA);
                let r_pcen = (r_mag / denom1 + PCEN_DELTA).powf(PCEN_R) - PCEN_DELTA.powf(PCEN_R);

                let bin_pcen_energy = (l_pcen + r_pcen) * 0.5;
                sum_pcen += bin_pcen_energy;

                let mag = l_mag + r_mag;
                if mag > 1e-9 {
                    let dir_bin = (r_mag - l_mag) / mag;
                    let weight = bin_pcen_energy;
                    weighted_dir_num += dir_bin * weight;
                    weighted_dir_den += weight;
                }
            }

            let count = (hi - lo).max(1) as f32;
            let band_energy = sum_pcen / count;
            let band_dir = if weighted_dir_den > 1e-9 {
                (weighted_dir_num / weighted_dir_den).clamp(-1.0, 1.0)
            } else {
                0.0
            };

            self.output.store_energy(b, band_energy);
            self.output.store_direction(b, band_dir);
        }

        cs0.pcen_init = true;
        cs1.pcen_init = true;
    }
}

fn start_audio(levels: Arc<BandLevels>) -> anyhow::Result<()> {
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or_else(|| anyhow::anyhow!("no default output device"))?;
    let config = device.default_output_config()?;
    let sample_rate = config.sample_rate().0 as f32;
    let channels = config.channels() as usize;
    let sample_format = config.sample_format();
    let stream_config: cpal::StreamConfig = config.into();

    let mut analyzer = Analyzer::new(sample_rate, channels, levels);
    let err_fn = |err| eprintln!("stream error: {err}");

    let stream = match sample_format {
        cpal::SampleFormat::F32 => device.build_input_stream(
            &stream_config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| analyzer.process(data),
            err_fn,
            None,
        )?,
        cpal::SampleFormat::I16 => {
            let scale = 1.0 / i16::MAX as f32;
            let mut buf: Vec<f32> = Vec::with_capacity(8192);
            device.build_input_stream(
                &stream_config,
                move |data: &[i16], _: &cpal::InputCallbackInfo| {
                    buf.clear();
                    buf.extend(data.iter().map(|&s| s as f32 * scale));
                    analyzer.process(&buf);
                },
                err_fn,
                None,
            )?
        }
        fmt => anyhow::bail!("unsupported sample format: {fmt:?}"),
    };

    stream.play()?;
    std::mem::forget(stream);
    Ok(())
}

// ───────── settings ─────────

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
struct HotkeyDef {
    enabled: bool,
    ctrl: bool,
    alt: bool,
    shift: bool,
    key: String, // "" = unset
}

impl Default for HotkeyDef {
    fn default() -> Self {
        Self {
            enabled: true,
            ctrl: true,
            alt: true,
            shift: false,
            key: "H".into(),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct Settings {
    pos_x: f32,
    pos_y: f32,
    width: f32,
    height: f32,
    opacity: f32,
    ball_size: f32,
    sensitivity: f32,
    show_band: [bool; 4],
    #[serde(default)]
    hotkey: HotkeyDef,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            pos_x: -1.0, // sentinel: center on first run
            pos_y: 16.0,
            width: 620.0,
            height: 116.0,
            opacity: 0.95,
            ball_size: 1.0,
            sensitivity: 0.7,
            show_band: [true; 4],
            hotkey: HotkeyDef::default(),
        }
    }
}

// ───────── hotkey helpers ─────────

fn hotkey_label(h: &HotkeyDef) -> String {
    if h.key.is_empty() {
        return "(none)".to_string();
    }
    let mut parts = Vec::new();
    if h.ctrl {
        parts.push("Ctrl");
    }
    if h.alt {
        parts.push("Alt");
    }
    if h.shift {
        parts.push("Shift");
    }
    let mut s = parts.join(" + ");
    if !s.is_empty() {
        s.push_str(" + ");
    }
    s.push_str(&h.key);
    s
}

fn build_hotkey(def: &HotkeyDef) -> Option<HotKey> {
    if !def.enabled {
        return None;
    }
    let code = key_str_to_code(&def.key)?;
    let mut mods = Modifiers::empty();
    if def.ctrl {
        mods |= Modifiers::CONTROL;
    }
    if def.alt {
        mods |= Modifiers::ALT;
    }
    if def.shift {
        mods |= Modifiers::SHIFT;
    }
    Some(HotKey::new(Some(mods), code))
}

fn key_str_to_code(s: &str) -> Option<Code> {
    match s.to_uppercase().as_str() {
        "A" => Some(Code::KeyA), "B" => Some(Code::KeyB), "C" => Some(Code::KeyC),
        "D" => Some(Code::KeyD), "E" => Some(Code::KeyE), "F" => Some(Code::KeyF),
        "G" => Some(Code::KeyG), "H" => Some(Code::KeyH), "I" => Some(Code::KeyI),
        "J" => Some(Code::KeyJ), "K" => Some(Code::KeyK), "L" => Some(Code::KeyL),
        "M" => Some(Code::KeyM), "N" => Some(Code::KeyN), "O" => Some(Code::KeyO),
        "P" => Some(Code::KeyP), "Q" => Some(Code::KeyQ), "R" => Some(Code::KeyR),
        "S" => Some(Code::KeyS), "T" => Some(Code::KeyT), "U" => Some(Code::KeyU),
        "V" => Some(Code::KeyV), "W" => Some(Code::KeyW), "X" => Some(Code::KeyX),
        "Y" => Some(Code::KeyY), "Z" => Some(Code::KeyZ),
        "0" => Some(Code::Digit0), "1" => Some(Code::Digit1), "2" => Some(Code::Digit2),
        "3" => Some(Code::Digit3), "4" => Some(Code::Digit4), "5" => Some(Code::Digit5),
        "6" => Some(Code::Digit6), "7" => Some(Code::Digit7), "8" => Some(Code::Digit8),
        "9" => Some(Code::Digit9),
        "F1" => Some(Code::F1), "F2" => Some(Code::F2), "F3" => Some(Code::F3),
        "F4" => Some(Code::F4), "F5" => Some(Code::F5), "F6" => Some(Code::F6),
        "F7" => Some(Code::F7), "F8" => Some(Code::F8), "F9" => Some(Code::F9),
        "F10" => Some(Code::F10), "F11" => Some(Code::F11), "F12" => Some(Code::F12),
        "SPACE" => Some(Code::Space),
        "ENTER" => Some(Code::Enter),
        "TAB" => Some(Code::Tab),
        "UP" => Some(Code::ArrowUp),
        "DOWN" => Some(Code::ArrowDown),
        "LEFT" => Some(Code::ArrowLeft),
        "RIGHT" => Some(Code::ArrowRight),
        "INSERT" => Some(Code::Insert),
        "DELETE" => Some(Code::Delete),
        "HOME" => Some(Code::Home),
        "END" => Some(Code::End),
        "PAGEUP" => Some(Code::PageUp),
        "PAGEDOWN" => Some(Code::PageDown),
        _ => None,
    }
}

fn egui_key_to_str(key: egui::Key) -> Option<&'static str> {
    use egui::Key as K;
    Some(match key {
        K::A => "A", K::B => "B", K::C => "C", K::D => "D", K::E => "E",
        K::F => "F", K::G => "G", K::H => "H", K::I => "I", K::J => "J",
        K::K => "K", K::L => "L", K::M => "M", K::N => "N", K::O => "O",
        K::P => "P", K::Q => "Q", K::R => "R", K::S => "S", K::T => "T",
        K::U => "U", K::V => "V", K::W => "W", K::X => "X", K::Y => "Y",
        K::Z => "Z",
        K::Num0 => "0", K::Num1 => "1", K::Num2 => "2", K::Num3 => "3",
        K::Num4 => "4", K::Num5 => "5", K::Num6 => "6", K::Num7 => "7",
        K::Num8 => "8", K::Num9 => "9",
        K::F1 => "F1", K::F2 => "F2", K::F3 => "F3", K::F4 => "F4",
        K::F5 => "F5", K::F6 => "F6", K::F7 => "F7", K::F8 => "F8",
        K::F9 => "F9", K::F10 => "F10", K::F11 => "F11", K::F12 => "F12",
        K::Space => "Space",
        K::Enter => "Enter",
        K::Tab => "Tab",
        K::ArrowUp => "Up",
        K::ArrowDown => "Down",
        K::ArrowLeft => "Left",
        K::ArrowRight => "Right",
        K::Insert => "Insert",
        K::Delete => "Delete",
        K::Home => "Home",
        K::End => "End",
        K::PageUp => "PageUp",
        K::PageDown => "PageDown",
        _ => return None,
    })
}

fn settings_path() -> PathBuf {
    let appdata = std::env::var("APPDATA").unwrap_or_else(|_| ".".into());
    let dir = PathBuf::from(appdata).join("sound_dir");
    let _ = std::fs::create_dir_all(&dir);
    dir.join("settings.json")
}

fn load_settings() -> Settings {
    let path = settings_path();
    if let Ok(data) = std::fs::read_to_string(&path) {
        if let Ok(s) = serde_json::from_str(&data) {
            return s;
        }
    }
    Settings::default()
}

fn save_settings(s: &Settings) {
    if let Ok(data) = serde_json::to_string_pretty(s) {
        let _ = std::fs::write(settings_path(), data);
    }
}

// ───────── tray icon ─────────

fn make_tray_rgba(size: u32) -> (Vec<u8>, u32, u32) {
    let mut img = vec![0u8; (size * size * 4) as usize];
    let colors = [
        [80u8, 170, 255, 255],
        [255, 110, 50, 255],
        [80, 240, 130, 255],
        [255, 230, 90, 255],
    ];
    let cx = size as f32 / 2.0;
    let cy = size as f32 / 2.0;
    let r_outer = size as f32 * 0.46;
    let ring = size as f32 * 0.18;
    let r_inner = r_outer - ring;
    for y in 0..size {
        for x in 0..size {
            let dx = x as f32 + 0.5 - cx;
            let dy = y as f32 + 0.5 - cy;
            let dist = (dx * dx + dy * dy).sqrt();
            let i = ((y * size + x) * 4) as usize;
            if dist <= r_outer {
                if dist >= r_inner {
                    let q = if dx < 0.0 && dy < 0.0 {
                        0
                    } else if dy < 0.0 {
                        1
                    } else if dx < 0.0 {
                        2
                    } else {
                        3
                    };
                    img[i..i + 4].copy_from_slice(&colors[q]);
                } else if dist <= r_inner * 0.55 {
                    img[i..i + 4].copy_from_slice(&[255, 255, 255, 255]);
                } else {
                    img[i..i + 4].copy_from_slice(&[18, 18, 22, 255]);
                }
            }
        }
    }
    (img, size, size)
}

struct TrayHandles {
    _tray: TrayIcon,
    edit_id: tray_icon::menu::MenuId,
    reset_id: tray_icon::menu::MenuId,
    quit_id: tray_icon::menu::MenuId,
}

fn setup_tray() -> anyhow::Result<TrayHandles> {
    let menu = Menu::new();
    let edit_item = MenuItem::new("Edit position", true, None);
    let reset_item = MenuItem::new("Reset to defaults", true, None);
    let quit_item = MenuItem::new("Quit", true, None);
    menu.append(&edit_item)?;
    menu.append(&reset_item)?;
    menu.append(&quit_item)?;

    let (rgba, w, h) = make_tray_rgba(32);
    let icon = tray_icon::Icon::from_rgba(rgba, w, h)
        .map_err(|e| anyhow::anyhow!("tray icon: {e}"))?;

    let tray = TrayIconBuilder::new()
        .with_menu(Box::new(menu))
        .with_tooltip("sound_dir")
        .with_icon(icon)
        .build()
        .map_err(|e| anyhow::anyhow!("build tray: {e}"))?;

    Ok(TrayHandles {
        _tray: tray,
        edit_id: edit_item.id().clone(),
        reset_id: reset_item.id().clone(),
        quit_id: quit_item.id().clone(),
    })
}

// ───────── direction → display mapping ─────────

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Minimal background — flat translucent dark rectangle. No border, no shadow,
/// no gradient. In edit mode shows L-shaped corner markers (so user can see what
/// they're moving) instead of a full frame.
fn paint_glass_strip(painter: &egui::Painter, rect: Rect, opacity: f32, edit: bool) {
    let opa = opacity.clamp(0.20, 1.0);
    let rounding = egui::Rounding::same(4.0);

    painter.rect_filled(
        rect,
        rounding,
        Color32::from_rgba_premultiplied(10, 11, 13, (170.0 * opa) as u8),
    );

    if edit {
        let len = 14.0;
        let stroke = egui::Stroke::new(
            1.5,
            Color32::from_rgba_premultiplied(255, 255, 255, 200),
        );
        let lt = rect.left_top();
        let rt = rect.right_top();
        let lb = rect.left_bottom();
        let rb = rect.right_bottom();
        painter.line_segment([lt, lt + Vec2::new(len, 0.0)], stroke);
        painter.line_segment([lt, lt + Vec2::new(0.0, len)], stroke);
        painter.line_segment([rt, rt + Vec2::new(-len, 0.0)], stroke);
        painter.line_segment([rt, rt + Vec2::new(0.0, len)], stroke);
        painter.line_segment([lb, lb + Vec2::new(len, 0.0)], stroke);
        painter.line_segment([lb, lb + Vec2::new(0.0, -len)], stroke);
        painter.line_segment([rb, rb + Vec2::new(-len, 0.0)], stroke);
        painter.line_segment([rb, rb + Vec2::new(0.0, -len)], stroke);
    }
}

// ───────── overlay UI ─────────

struct OverlayApp {
    levels: Arc<BandLevels>,
    smooth_dir: [f32; 4],
    smooth_energy: [f32; 4],
    auto_max: f32,
    last_frame: Instant,
    frame_count: u64,

    settings: Settings,
    edit_mode: bool,
    needs_save: bool,
    visible: bool,
    awaiting_hotkey: bool,

    tray: TrayHandles,
    hotkey_mgr: GlobalHotKeyManager,
    current_hotkey: Option<HotKey>,
}

impl OverlayApp {
    fn new(
        levels: Arc<BandLevels>,
        settings: Settings,
        tray: TrayHandles,
        hotkey_mgr: GlobalHotKeyManager,
    ) -> Self {
        let mut app = Self {
            levels,
            smooth_dir: [0.0; 4],
            smooth_energy: [0.0; 4],
            auto_max: 0.5,
            last_frame: Instant::now(),
            frame_count: 0,
            settings,
            edit_mode: false,
            needs_save: false,
            visible: true,
            awaiting_hotkey: false,
            tray,
            hotkey_mgr,
            current_hotkey: None,
        };
        app.refresh_hotkey();
        app
    }

    fn set_edit_mode(&mut self, ctx: &egui::Context, on: bool) {
        if self.edit_mode == on {
            return;
        }
        self.edit_mode = on;
        ctx.send_viewport_cmd(egui::ViewportCommand::MousePassthrough(!on));
        if !on {
            self.awaiting_hotkey = false;
            self.flush_save();
        }
    }

    fn flush_save(&mut self) {
        if self.needs_save {
            save_settings(&self.settings);
            self.needs_save = false;
        }
    }

    fn refresh_hotkey(&mut self) {
        if let Some(hk) = self.current_hotkey.take() {
            let _ = self.hotkey_mgr.unregister(hk);
        }
        if let Some(hk) = build_hotkey(&self.settings.hotkey) {
            if self.hotkey_mgr.register(hk).is_ok() {
                self.current_hotkey = Some(hk);
            }
        }
    }
}

impl eframe::App for OverlayApp {
    fn clear_color(&self, _visuals: &egui::Visuals) -> [f32; 4] {
        [0.0, 0.0, 0.0, 0.0]
    }

    fn on_exit(&mut self) {
        self.flush_save();
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.request_repaint();

        // 1. Tray menu events
        while let Ok(ev) = MenuEvent::receiver().try_recv() {
            if ev.id == self.tray.edit_id {
                self.set_edit_mode(ctx, !self.edit_mode);
            } else if ev.id == self.tray.reset_id {
                self.settings = Settings::default();
                self.settings.pos_x = -1.0;
                self.needs_save = true;
                self.refresh_hotkey();
                self.flush_save();
            } else if ev.id == self.tray.quit_id {
                self.flush_save();
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
        }

        // 1a. Global hotkey events (toggle visibility)
        while let Ok(ev) = GlobalHotKeyEvent::receiver().try_recv() {
            if ev.state == HotKeyState::Pressed {
                if let Some(ref hk) = self.current_hotkey {
                    if ev.id == hk.id() {
                        self.visible = !self.visible;
                    }
                }
            }
        }

        // 1b. Hotkey capture mode (user clicked "set hotkey" in settings)
        if self.awaiting_hotkey {
            let captured: Option<Result<(egui::Modifiers, &'static str), ()>> =
                ctx.input(|i| {
                    for ev in &i.events {
                        if let egui::Event::Key {
                            key,
                            pressed: true,
                            modifiers,
                            ..
                        } = ev
                        {
                            if *key == egui::Key::Escape {
                                return Some(Err(()));
                            }
                            // require at least one modifier so we don't grab plain letters
                            if !(modifiers.ctrl || modifiers.alt || modifiers.shift) {
                                continue;
                            }
                            if let Some(s) = egui_key_to_str(*key) {
                                return Some(Ok((*modifiers, s)));
                            }
                        }
                    }
                    None
                });
            match captured {
                Some(Err(())) => {
                    self.awaiting_hotkey = false;
                }
                Some(Ok((mods, key_str))) => {
                    self.settings.hotkey.ctrl = mods.ctrl;
                    self.settings.hotkey.alt = mods.alt;
                    self.settings.hotkey.shift = mods.shift;
                    self.settings.hotkey.key = key_str.to_string();
                    self.settings.hotkey.enabled = true;
                    self.needs_save = true;
                    self.awaiting_hotkey = false;
                    self.refresh_hotkey();
                }
                None => {}
            }
        }

        // 2. Initial passthrough (delayed by a few frames so window can mount cleanly)
        if self.frame_count == 5 {
            ctx.send_viewport_cmd(egui::ViewportCommand::MousePassthrough(!self.edit_mode));
        }
        self.frame_count += 1;

        // 3. dt + smoothing
        let now = Instant::now();
        let dt = now.duration_since(self.last_frame).as_secs_f32().min(0.1);
        self.last_frame = now;

        // sensitivity ∈ [0,1] maps response speed: 0 → slow & smooth, 1 → instant.
        let s = self.settings.sensitivity.clamp(0.0, 1.0);
        let dir_tau = 0.060 - 0.050 * s; // 60ms..10ms
        let attack_tau = 0.045 - 0.035 * s; // 45ms..10ms
        let release_tau = 0.220 - 0.140 * s; // 220ms..80ms
        let dir_alpha = 1.0 - (-dt / dir_tau).exp();
        let energy_attack = 1.0 - (-dt / attack_tau).exp();
        let energy_release = 1.0 - (-dt / release_tau).exp();

        for b in 0..4 {
            let energy = self.levels.load_energy(b);
            let direction = self.levels.load_direction(b);
            let alpha = if energy > self.smooth_energy[b] {
                energy_attack
            } else {
                energy_release
            };
            self.smooth_energy[b] += alpha * (energy - self.smooth_energy[b]);
            self.smooth_dir[b] += dir_alpha * (direction - self.smooth_dir[b]);
        }

        // 4. Auto-scale based on peak energy
        let mut peak = 0.0f32;
        for b in 0..4 {
            peak = peak.max(self.smooth_energy[b]);
        }
        let target = peak.max(0.15);
        let auto_attack = 1.0 - (-dt / 0.10).exp();
        let auto_release = 1.0 - (-dt / 6.0).exp();
        if target > self.auto_max {
            self.auto_max += auto_attack * (target - self.auto_max);
        } else {
            self.auto_max += auto_release * (target - self.auto_max);
        }

        // 5. Render
        let smooth_dir = self.smooth_dir;
        let smooth_energy = self.smooth_energy;
        let auto_max = self.auto_max;
        let edit_mode = self.edit_mode;
        let visible = self.visible || self.edit_mode;
        let awaiting_hotkey = self.awaiting_hotkey;
        let mut settings = self.settings.clone();
        let mut request_lock = false;
        let mut request_capture = false;
        let mut request_hotkey_refresh = false;

        egui::CentralPanel::default()
            .frame(egui::Frame::none().fill(Color32::TRANSPARENT))
            .show(ctx, |ui| {
                let screen = ui.max_rect();

                if settings.pos_x < 0.0 {
                    settings.pos_x = (screen.width() - settings.width) * 0.5;
                }
                settings.width = settings.width.clamp(280.0, 1600.0);
                settings.height = settings.height.clamp(60.0, 240.0);
                settings.pos_x = settings
                    .pos_x
                    .clamp(0.0, (screen.width() - settings.width).max(0.0));
                settings.pos_y = settings
                    .pos_y
                    .clamp(0.0, (screen.height() - settings.height - 24.0).max(0.0));

                if !visible {
                    return; // hidden via hotkey — render nothing
                }

                let strip_h = settings.height;
                let strip_rect = Rect::from_min_size(
                    Pos2::new(settings.pos_x, settings.pos_y),
                    Vec2::new(settings.width, strip_h),
                );

                let painter = ui.painter();

                // ---------- background panel ----------
                paint_glass_strip(painter, strip_rect, settings.opacity, edit_mode);

                // ---------- center reference line ----------
                let center_x = strip_rect.center().x;
                painter.line_segment(
                    [
                        Pos2::new(center_x, strip_rect.top() + 10.0),
                        Pos2::new(center_x, strip_rect.bottom() - 10.0),
                    ],
                    egui::Stroke::new(
                        1.0,
                        Color32::from_rgba_premultiplied(255, 255, 255, 55),
                    ),
                );

                // ---------- tracks + balls ----------
                let visible_count = settings.show_band.iter().filter(|v| **v).count().max(1);
                let inner_top_pad = 12.0;
                let inner_bot_pad = 12.0;
                let inner_h = (strip_h - inner_top_pad - inner_bot_pad).max(16.0);
                let track_pad = 4.0;
                let total_pad = (visible_count.saturating_sub(1)) as f32 * track_pad;
                let track_h =
                    ((inner_h - total_pad) / visible_count as f32).clamp(8.0, 36.0);

                let outer_pad = 18.0; // safe margin so balls never escape the panel
                let track_w = (settings.width - outer_pad * 2.0).max(80.0);
                let track_left = strip_rect.left() + outer_pad;
                let mut track_y = strip_rect.top() + inner_top_pad;

                let max_ball_r = (track_h * 0.45).min(outer_pad - 2.0).max(2.0);

                for b in 0..4 {
                    if !settings.show_band[b] {
                        continue;
                    }
                    let base = band_color(b);
                    let line_y = track_y + track_h * 0.5;

                    // very thin, very dim track line (just enough to imply orientation)
                    painter.line_segment(
                        [
                            Pos2::new(track_left, line_y),
                            Pos2::new(track_left + track_w, line_y),
                        ],
                        egui::Stroke::new(1.0, with_alpha(base, 32)),
                    );

                    let dir = smooth_dir[b].clamp(-1.0, 1.0);
                    let energy_norm = (smooth_energy[b] / auto_max.max(1e-3)).clamp(0.0, 1.0);

                    if energy_norm > 0.02 {
                        let base_r = 3.5 + 6.0 * energy_norm.sqrt();
                        let r = (base_r * settings.ball_size).min(max_ball_r);
                        let raw_x = track_left + track_w * (0.5 + 0.5 * dir);
                        let ball_x = raw_x.clamp(
                            strip_rect.left() + r + 2.0,
                            strip_rect.right() - r - 2.0,
                        );
                        let alpha = ((180.0 + 75.0 * energy_norm) * settings.opacity)
                            .clamp(0.0, 255.0) as u8;

                        // single solid circle. nothing else.
                        painter.circle_filled(
                            Pos2::new(ball_x, line_y),
                            r,
                            with_alpha(base, alpha),
                        );
                    }

                    track_y += track_h + track_pad;
                }

                // ---------- edit mode interactions ----------
                if edit_mode {
                    let drag_resp = ui.interact(
                        strip_rect,
                        ui.id().with("strip_drag"),
                        egui::Sense::click_and_drag(),
                    );
                    if drag_resp.dragged() {
                        let delta = drag_resp.drag_delta();
                        settings.pos_x += delta.x;
                        settings.pos_y += delta.y;
                    }

                    // resize handle in bottom-right corner
                    let handle_size = 16.0;
                    let handle_rect = Rect::from_min_size(
                        Pos2::new(
                            strip_rect.right() - handle_size - 2.0,
                            strip_rect.bottom() - handle_size - 2.0,
                        ),
                        Vec2::new(handle_size, handle_size),
                    );
                    // diagonal lines to suggest "resize" affordance
                    let hc = Color32::from_rgba_premultiplied(255, 255, 255, 200);
                    for off in [0.0f32, 4.0, 8.0] {
                        painter.line_segment(
                            [
                                Pos2::new(
                                    handle_rect.right() - off - 1.0,
                                    handle_rect.bottom() - 1.0,
                                ),
                                Pos2::new(
                                    handle_rect.right() - 1.0,
                                    handle_rect.bottom() - off - 1.0,
                                ),
                            ],
                            egui::Stroke::new(1.5, hc),
                        );
                    }
                    let resize_resp = ui.interact(
                        handle_rect,
                        ui.id().with("resize"),
                        egui::Sense::drag(),
                    );
                    if resize_resp.dragged() {
                        let d = resize_resp.drag_delta();
                        settings.width = (settings.width + d.x).clamp(280.0, 1600.0);
                        settings.height = (settings.height + d.y).clamp(60.0, 240.0);
                    }
                }
            });

        // 6. Settings panel (only in edit mode)
        if self.edit_mode {
            egui::Window::new("settings")
                .collapsible(false)
                .resizable(false)
                .title_bar(false)
                .frame(egui::Frame {
                    fill: Color32::from_rgba_premultiplied(20, 20, 24, 245),
                    stroke: egui::Stroke::new(
                        1.0,
                        Color32::from_rgba_premultiplied(110, 110, 120, 255),
                    ),
                    inner_margin: egui::Margin::same(12.0),
                    rounding: egui::Rounding::same(2.0),
                    ..Default::default()
                })
                .fixed_pos(Pos2::new(
                    settings.pos_x,
                    settings.pos_y + settings.height + 12.0,
                ))
                .show(ctx, |ui| {
                    ui.set_min_width(260.0);
                    ui.set_max_width(300.0);

                    let mono = |s: &str| egui::RichText::new(s).monospace().size(11.0);
                    let header = |s: &str| {
                        egui::RichText::new(s)
                            .monospace()
                            .size(10.0)
                            .color(Color32::from_rgb(120, 130, 145))
                    };
                    let label = |s: &str| {
                        egui::RichText::new(s)
                            .monospace()
                            .size(11.0)
                            .color(Color32::from_rgb(170, 175, 185))
                    };

                    ui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new("sound_dir")
                                .monospace()
                                .size(12.0)
                                .color(Color32::from_rgb(230, 235, 245)),
                        );
                        ui.label(
                            egui::RichText::new("/ edit")
                                .monospace()
                                .size(11.0)
                                .color(Color32::from_rgb(120, 130, 145)),
                        );
                    });

                    ui.add_space(8.0);

                    // ─── geometry ───
                    ui.label(header("GEOMETRY"));
                    ui.add_space(2.0);
                    egui::Grid::new("geom_grid")
                        .num_columns(3)
                        .spacing([8.0, 4.0])
                        .show(ui, |ui| {
                            ui.label(label("width"));
                            ui.add(
                                egui::Slider::new(&mut settings.width, 280.0..=1400.0)
                                    .show_value(false),
                            );
                            ui.label(mono(&format!("{:>4.0}", settings.width)));
                            ui.end_row();

                            ui.label(label("height"));
                            ui.add(
                                egui::Slider::new(&mut settings.height, 60.0..=240.0)
                                    .show_value(false),
                            );
                            ui.label(mono(&format!("{:>4.0}", settings.height)));
                            ui.end_row();
                        });

                    ui.add_space(8.0);

                    // ─── appearance ───
                    ui.label(header("APPEARANCE"));
                    ui.add_space(2.0);
                    egui::Grid::new("app_grid")
                        .num_columns(3)
                        .spacing([8.0, 4.0])
                        .show(ui, |ui| {
                            ui.label(label("opacity"));
                            ui.add(
                                egui::Slider::new(&mut settings.opacity, 0.20..=1.00)
                                    .show_value(false),
                            );
                            ui.label(mono(&format!("{:>3.0}%", settings.opacity * 100.0)));
                            ui.end_row();

                            ui.label(label("balls"));
                            ui.add(
                                egui::Slider::new(&mut settings.ball_size, 0.50..=2.00)
                                    .show_value(false),
                            );
                            ui.label(mono(&format!("{:>3.0}%", settings.ball_size * 100.0)));
                            ui.end_row();
                        });

                    ui.add_space(8.0);

                    // ─── response ───
                    ui.label(header("RESPONSE"));
                    ui.add_space(2.0);
                    egui::Grid::new("resp_grid")
                        .num_columns(3)
                        .spacing([8.0, 4.0])
                        .show(ui, |ui| {
                            ui.label(label("speed"));
                            ui.add(
                                egui::Slider::new(&mut settings.sensitivity, 0.0..=1.0)
                                    .show_value(false),
                            );
                            ui.label(mono(&format!("{:>3.0}%", settings.sensitivity * 100.0)));
                            ui.end_row();
                        });

                    ui.add_space(8.0);

                    // ─── bands ───
                    ui.label(header("BANDS"));
                    ui.add_space(2.0);
                    let names = ["ROAR    20-200 Hz", "STEPS  200-2k Hz", "SHOTS  2k-5k Hz", "CLINK  5k-10k Hz"];
                    for b in 0..4 {
                        ui.horizontal(|ui| {
                            ui.checkbox(&mut settings.show_band[b], "");
                            ui.label(
                                egui::RichText::new(names[b])
                                    .monospace()
                                    .size(11.0)
                                    .color(band_color(b)),
                            );
                        });
                    }

                    ui.add_space(8.0);

                    // ─── hotkey ───
                    ui.label(header("HOTKEY"));
                    ui.add_space(2.0);
                    ui.horizontal(|ui| {
                        ui.label(label("hide / show"));
                        let btn_text = if awaiting_hotkey {
                            "press keys… (Esc to cancel)".to_string()
                        } else {
                            hotkey_label(&settings.hotkey)
                        };
                        let btn_color = if awaiting_hotkey {
                            Color32::from_rgb(255, 200, 100)
                        } else {
                            Color32::from_rgb(220, 220, 230)
                        };
                        if ui
                            .add(
                                egui::Button::new(
                                    egui::RichText::new(btn_text)
                                        .monospace()
                                        .size(11.0)
                                        .color(btn_color),
                                )
                                .min_size(Vec2::new(160.0, 22.0)),
                            )
                            .clicked()
                            && !awaiting_hotkey
                        {
                            request_capture = true;
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.add_space(4.0);
                        let prev = settings.hotkey.enabled;
                        ui.checkbox(&mut settings.hotkey.enabled, mono("enabled"));
                        if prev != settings.hotkey.enabled {
                            request_hotkey_refresh = true;
                        }
                    });

                    ui.add_space(10.0);
                    ui.separator();
                    ui.add_space(4.0);

                    ui.horizontal(|ui| {
                        if ui
                            .add(
                                egui::Button::new(mono("save & lock"))
                                    .min_size(Vec2::new(110.0, 24.0)),
                            )
                            .clicked()
                        {
                            request_lock = true;
                        }
                        if ui
                            .add(
                                egui::Button::new(mono("reset"))
                                    .min_size(Vec2::new(64.0, 24.0)),
                            )
                            .clicked()
                        {
                            settings = Settings::default();
                            settings.pos_x = -1.0;
                        }
                    });

                    ui.add_space(4.0);
                    ui.label(
                        egui::RichText::new(
                            "drag strip · ↘ corner to resize · auto-saved",
                        )
                        .monospace()
                        .size(9.5)
                        .color(Color32::from_rgb(110, 115, 125)),
                    );
                });
        }

        // 7. Persist any settings changes from this frame
        if !settings_eq(&self.settings, &settings) {
            self.settings = settings;
            self.needs_save = true;
        }
        if request_capture {
            self.awaiting_hotkey = true;
        }
        if request_hotkey_refresh {
            self.refresh_hotkey();
        }
        if request_lock {
            self.set_edit_mode(ctx, false);
        }
    }
}

fn settings_eq(a: &Settings, b: &Settings) -> bool {
    (a.pos_x - b.pos_x).abs() < 0.001
        && (a.pos_y - b.pos_y).abs() < 0.001
        && (a.width - b.width).abs() < 0.001
        && (a.height - b.height).abs() < 0.001
        && (a.opacity - b.opacity).abs() < 0.001
        && (a.ball_size - b.ball_size).abs() < 0.001
        && (a.sensitivity - b.sensitivity).abs() < 0.001
        && a.show_band == b.show_band
        && a.hotkey == b.hotkey
}

fn with_alpha(c: Color32, a: u8) -> Color32 {
    let r = ((c.r() as u32 * a as u32) / 255) as u8;
    let g = ((c.g() as u32 * a as u32) / 255) as u8;
    let b = ((c.b() as u32 * a as u32) / 255) as u8;
    Color32::from_rgba_premultiplied(r, g, b, a)
}

// ───────── main ─────────

fn main() -> anyhow::Result<()> {
    let levels = Arc::new(BandLevels::new());
    start_audio(levels.clone())?;

    let settings = load_settings();
    let tray = setup_tray()?;
    let hotkey_mgr = GlobalHotKeyManager::new()
        .map_err(|e| anyhow::anyhow!("global-hotkey init: {e}"))?;

    let displays = display_info::DisplayInfo::all().unwrap_or_default();
    let primary = displays
        .iter()
        .find(|d| d.is_primary)
        .or_else(|| displays.first());
    let (size_w, size_h) = match primary {
        Some(d) => {
            let scale = if d.scale_factor > 0.0 {
                d.scale_factor
            } else {
                1.0
            };
            ((d.width as f32) / scale, (d.height as f32) / scale)
        }
        None => (1920.0, 1080.0),
    };

    let viewport = egui::ViewportBuilder::default()
        .with_transparent(true)
        .with_decorations(false)
        .with_resizable(false)
        .with_always_on_top()
        .with_active(false)
        .with_taskbar(false)
        .with_title("sound_dir")
        .with_inner_size([size_w, size_h])
        .with_position([0.0, 0.0]);

    let options = eframe::NativeOptions {
        viewport,
        vsync: true,
        ..Default::default()
    };

    let levels_app = levels.clone();
    eframe::run_native(
        "sound_dir",
        options,
        Box::new(move |_cc| {
            Box::new(OverlayApp::new(levels_app, settings, tray, hotkey_mgr))
        }),
    )
    .map_err(|e| anyhow::anyhow!("eframe error: {e}"))?;

    Ok(())
}
