use std::env;
use std::path::PathBuf;

fn generate_rgba(size: u32) -> Vec<u8> {
    let mut img = vec![0u8; (size * size * 4) as usize];
    let colors = [
        [80u8, 170, 255, 255],
        [255, 110, 50, 255],
        [80, 240, 130, 255],
        [255, 230, 90, 255],
    ];
    let cx = size as f32 / 2.0;
    let cy = size as f32 / 2.0;
    let r_outer = (size as f32 * 0.46).max(8.0);
    let ring_thickness = (size as f32 * 0.18).max(4.0);
    let r_inner = r_outer - ring_thickness;
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
                    } else if dx >= 0.0 && dy < 0.0 {
                        1
                    } else if dx < 0.0 && dy >= 0.0 {
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
    img
}

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let icon_path = out_dir.join("icon.ico");

    {
        use ico::{IconDir, IconDirEntry, IconImage, ResourceType};
        let mut dir = IconDir::new(ResourceType::Icon);
        for &size in &[16u32, 32, 48, 64, 128, 256] {
            let rgba = generate_rgba(size);
            let img = IconImage::from_rgba_data(size, size, rgba);
            dir.add_entry(IconDirEntry::encode(&img).unwrap());
        }
        let file = std::fs::File::create(&icon_path).unwrap();
        dir.write(file).unwrap();
    }

    if cfg!(target_os = "windows") {
        let mut res = winresource::WindowsResource::new();
        res.set_icon(icon_path.to_str().unwrap());
        if let Err(e) = res.compile() {
            eprintln!("warning: failed to embed icon: {}", e);
        }
    }

    println!("cargo:rerun-if-changed=build.rs");
}
