//! Radiance RGBE panorama decoding. Stored linear RGB values are preserved.
use glam::Vec3;
use govno_core::BoxError;
use image::{ImageDecoder, codecs::hdr::HdrDecoder};
use std::{
    fs::File,
    io::{BufReader, Read},
    path::Path,
};

/// Bounds decoded RGB pixels before allocation (up to an 8192x4096 panorama).
pub const MAX_HDR_PIXELS: u64 = 32 * 1024 * 1024;
pub struct HdrPanorama {
    pub width: u32,
    pub height: u32,
    pub radiance: Vec<Vec3>,
    /// Header metadata only; no inverse exposure is silently applied to pixels.
    pub exposure: Option<f32>,
    /// Header metadata only; no color correction or primary conversion is applied.
    pub color_correction: Option<Vec3>,
}
pub fn load_hdr(path: impl AsRef<Path>) -> Result<HdrPanorama, BoxError> {
    let path = path.as_ref();
    let result = (|| decode_hdr(BufReader::new(File::open(path)?)))();
    result
        .map_err(|error| format!("failed to load HDR panorama {}: {error}", path.display()).into())
}
pub fn decode_hdr(reader: impl Read) -> Result<HdrPanorama, BoxError> {
    let decoder = HdrDecoder::new(reader)?;
    let metadata = decoder.metadata();
    let (width, height) = decoder.dimensions();
    let pixels = u64::from(width) * u64::from(height);
    if width < 4 || height < 2 || pixels > MAX_HDR_PIXELS {
        return Err(
            "HDR panorama needs width >= 4, height >= 2 and at most 33554432 pixels".into(),
        );
    }
    let correction = metadata
        .color_correction
        .map(|(r, g, b)| Vec3::new(r, g, b));
    if metadata
        .exposure
        .is_some_and(|e| !e.is_finite() || e <= 0.0)
        || correction.is_some_and(|c| !c.is_finite() || c.min_element() <= 0.0)
    {
        return Err("HDR exposure/color correction metadata must be finite and positive".into());
    }
    let mut bytes = vec![0; usize::try_from(decoder.total_bytes())?];
    decoder.read_image(&mut bytes)?;
    let mut radiance = Vec::with_capacity(pixels as usize);
    for rgb in bytes.as_chunks::<12>().0 {
        let value = Vec3::new(
            f32::from_ne_bytes(rgb[0..4].try_into()?),
            f32::from_ne_bytes(rgb[4..8].try_into()?),
            f32::from_ne_bytes(rgb[8..12].try_into()?),
        );
        if !value.is_finite() || value.min_element() < 0.0 {
            return Err("HDR panorama contains invalid linear radiance".into());
        }
        radiance.push(value);
    }
    Ok(HdrPanorama {
        width,
        height,
        radiance,
        exposure: metadata.exposure,
        color_correction: correction,
    })
}
