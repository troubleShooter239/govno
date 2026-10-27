//! Reproducible authored EXR panorama fixture.
use image::{ExtendedColorType, ImageEncoder, codecs::openexr::OpenExrEncoder};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).ok_or("output EXR path required")?;
    let pixels: Vec<f32> = (0..4)
        .flat_map(|y| {
            (0..8).flat_map(move |x| {
                if y == 0 && x < 2 {
                    [4.0, 2.0, 0.5]
                } else {
                    [0.1 + 0.1 * x as f32, 0.2, 0.1 + 0.15 * y as f32]
                }
            })
        })
        .collect();
    let bytes: Vec<u8> = pixels.iter().flat_map(|v| v.to_ne_bytes()).collect();
    OpenExrEncoder::new(std::io::BufWriter::new(std::fs::File::create(path)?)).write_image(
        &bytes,
        8,
        4,
        ExtendedColorType::Rgb32F,
    )?;
    Ok(())
}
