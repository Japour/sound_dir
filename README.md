# sound_dir

A directional sound visualizer for deaf and hard-of-hearing gamers, built in Rust.

I'm deaf with a cochlear implant on my left ear only. In games like PUBG, where survival depends on knowing which direction shots and footsteps come from, I'm at a permanent disadvantage — I hear sounds, but my brain can't compute direction from a single ear.

This program watches the system audio output, analyzes left/right balance per frequency band, and shows colored dots on a small strip at the top of the screen. Each dot drifts left when sound comes from the left, right when it comes from the right. Different colors for different sound types: blue for engines, green for footsteps, orange for gunshots, yellow for metallic clicks (grenade pins, glass).

It runs on top of any game without injecting into it.

## Screenshot

*Coming soon — running screenshots from PUBG / CS / Apex.*

## Features

- **Listens to system audio output** via WASAPI loopback, no game injection
- **Per-band direction estimation** weighted by [PCEN](https://arxiv.org/abs/1607.05666) energy — engine rumble doesn't drown out the gunshot
- **Configurable overlay** — drag to move, drag to resize, opacity / ball size / response speed all adjustable from the settings panel
- **System tray** — clean quit, edit mode, reset to defaults
- **Persistent settings** — saved to `%APPDATA%/sound_dir/settings.json`
- **Minimal CPU and RAM** — ~1–3 % CPU, ~50 MB RAM, ~9 MB binary
- **Low latency** — ~50–80 ms end-to-end from sound to dot moving

## How it works (one minute)

1. Capture system audio via WASAPI loopback (the same way OBS or Discord do)
2. FFT — 1024 samples at 48 kHz, hop of 256 samples (~190 Hz update rate)
3. PCEN per channel and per bin — automatically suppresses stationary noise (engines, ambient music) and amplifies transients (gunshots, footsteps)
4. Per-bin direction estimate weighted by PCEN energy. Stationary background contributes ~zero weight, transient events dominate the direction
5. Aggregate into 4 frequency bands: ROAR (20–200 Hz), STEPS (200–2000 Hz), SHOTS (2–5 kHz), CLINK (5–10 kHz)
6. Render as colored dots on a transparent always-on-top window using `egui` + `wgpu`

## Anti-cheat compatibility

The program only listens to the system audio output (the same audio your speakers receive). It does **not**:

- Read game memory
- Inject DLLs into the game
- Hook DirectX / Vulkan inside the game

It behaves the same way as Discord overlay, OBS, or NVIDIA ShadowPlay — listens to your audio, draws on your screen. BattlEye / EAC / Vanguard don't flag this.

> [!IMPORTANT]
> The game must be in **Windowed Fullscreen / Borderless** mode. In Exclusive Fullscreen, no overlay can show — that's a Windows DirectX limitation, not a sound_dir one. Most pros play in Borderless anyway because it's the only way to use Discord, OBS, or quick alt-tab.

## Install

### Pre-built binary

A release binary will appear in [Releases](../../releases) once the project stabilizes. For now: build from source.

### Build from source

You need [Rust](https://rustup.rs/) installed (1.75 or newer recommended). Then:

```bash
git clone https://github.com/Japour/sound_dir
cd sound_dir
cargo build --release
```

The binary will be at `target/release/sound_dir.exe` (Windows) or `target/release/sound_dir` (other platforms).

### Platform support

| OS | Status | Notes |
|---|---|---|
| Windows 10/11 | ✅ Tested | Primary target |

Currently I only develop and test on Windows. Help porting to macOS / Linux is welcome.

## Usage

1. Run `sound_dir.exe`. A desktop shortcut helps.
2. A small icon appears in the system tray (bottom-right corner near the clock). On Windows 11 it may be hidden behind the `^` arrow — drag it out into the always-visible area.
3. The compass strip appears at the top center of your screen, ready to react to sound.
4. **Right-click the tray icon** for the menu:
   - **Edit position** — enter edit mode
   - **Reset to defaults** — bring back default position / size
   - **Quit** — close cleanly

### Edit mode

In edit mode the strip becomes interactive:

- **Drag the strip** anywhere — moves it
- **Drag the corner handle** (bottom-right) — resizes width and height
- A **settings panel** appears below with:
  - Width / height sliders
  - Opacity slider
  - Ball size slider
  - Response speed slider (0 % = smooth and slow, 100 % = instant)
  - Per-band visibility checkboxes (hide ROAR if it bothers you)
- **Save & lock** exits edit mode and persists settings

Settings are saved automatically to `%APPDATA%/sound_dir/settings.json`.

## Tech stack

- **[Rust](https://www.rust-lang.org/)** — fast, safe, and a single binary
- **[`cpal`](https://crates.io/crates/cpal)** — cross-platform audio capture (WASAPI loopback on Windows)
- **[`realfft`](https://crates.io/crates/realfft)** — fast real-input FFT
- **[`eframe`](https://crates.io/crates/eframe)** (egui + wgpu) — transparent overlay with mouse passthrough
- **[`tray-icon`](https://crates.io/crates/tray-icon)** — system tray
- **[`serde`](https://crates.io/crates/serde) / [`serde_json`](https://crates.io/crates/serde_json)** — settings persistence

## Roadmap

Things I want to add when I have time. PRs welcome.

- [ ] **Onset detection** — make balls flash discretely on transients instead of being continuous
- [ ] **Hotkeys** — global shortcuts for edit mode and visibility toggle
- [ ] **Multi-source DOA** — detect multiple simultaneous directional sources
- [ ] **Sound classification** — small ML model (YAMNet?) to label sounds (footstep / gunshot / vehicle / explosion)
- [ ] **Per-game profiles** — quick-switch presets (PUBG / CSGO / Fortnite / Music)
- [ ] **Phase-based localization** for low frequencies
- [ ] **Real backdrop blur** via DWM Acrylic on Windows 11
- [ ] **macOS / Linux** — proper documentation and testing
- [ ] **Custom audio device selection** — currently uses default output
- [ ] **Front / back differentiation** (only possible with HRTF inversion — hard)

## Contributing

If you're a deaf or hard-of-hearing gamer who'd find this useful, a DSP enthusiast who wants to improve the math, or a Rust dev who wants to clean up the code — pull requests, issues, ideas all welcome. This started as a personal accessibility tool that worked well enough to share. I'd love to see it grow with the community.

When opening a PR:

1. Run `cargo fmt`
2. Make sure `cargo clippy --release` is clean
3. If you change the math (PCEN, direction estimation, smoothing), document the change in ARCHITECTURE.md

## License

[MIT](LICENSE). Use it, fork it, modify it, ship it. I just want deaf gamers to have a fair shot.

## Acknowledgments

- [Audio Radar](https://audioradar.com/) — commercial hardware product for deaf gamers, proof that the idea works
- [Fortnite Sound Visualizer](https://accessibility-labs.com/feature-highlight-fortnites-sound-visualizer/) — proof that visual sound indication is real accessibility, not a gimmick
- **Wang et al. (2017)**, [*Trainable Frontend For Robust and Far-Field Keyword Spotting*](https://arxiv.org/abs/1607.05666) — the PCEN paper that powers the "ignore the engine, hear the gunshot" magic
- The Rust audio community for [`cpal`](https://github.com/RustAudio/cpal) and [`realfft`](https://github.com/HEnquist/realfft)
- The [egui](https://github.com/emilk/egui) project for making transparent overlays approachable

---

Built by a deaf gamer for deaf gamers. If this helps you, open an issue and tell me — it'll make my day.
