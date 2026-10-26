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
#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    #[test]
    fn planar_run_packets_decode_without_channel_or_row_swapping() {
        let mut data = b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 2 +X 8\n".to_vec();
        for exponent in [130, 129] {
            data.extend_from_slice(&[2, 2, 0, 8]);
            for value in [128, 64, 32, exponent] {
                data.extend_from_slice(&[136, value]);
            }
        }
        let map = decode_hdr(Cursor::new(data)).unwrap();
        assert_eq!(map.radiance[0], Vec3::new(2.0, 1.0, 0.5));
        assert_eq!(map.radiance[7], map.radiance[0]);
        assert_eq!(map.radiance[8], Vec3::new(1.0, 0.5, 0.25));
    }
    #[test]
    fn fixture_preserves_hdr_energy_orientation_and_header_metadata() {
        let map = decode_hdr(Cursor::new(include_bytes!("../tests/fixtures/studio.hdr"))).unwrap();
        assert_eq!((map.width, map.height), (8, 4));
        assert_eq!(map.exposure, Some(2.0));
        assert_eq!(map.color_correction, Some(Vec3::new(1.0, 2.0, 1.0)));
        // Authored top-left white light is 4.0, not display-encoded or normalized.
        assert_eq!(map.radiance[0], Vec3::splat(4.0));
        assert_eq!(map.radiance[7], Vec3::new(0.25, 0.5, 2.0));
        assert_eq!(map.radiance[24], Vec3::new(0.25, 0.125, 0.0625));
        let cooked = govno_render::SpecularEnvironment::from_equirectangular(
            map.width,
            map.height,
            &map.radiance,
        )
        .unwrap();
        assert!(cooked.diffuse().evaluate(Vec3::Y).max_element() > 1.0);
    }
    #[test]
    fn uncompressed_rgbe_is_supported_without_clamping() {
        let mut data = b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 2 +X 4\n".to_vec();
        for _ in 0..8 {
            data.extend_from_slice(&[128, 64, 32, 130]);
        }
        let map = decode_hdr(Cursor::new(data)).unwrap();
        assert_eq!(map.radiance, vec![Vec3::new(2.0, 1.0, 0.5); 8]);
        assert_eq!(map.exposure, None);
    }
    #[test]
    fn truncated_unsupported_and_oversized_panorama_inputs_fail() {
        assert!(decode_hdr(Cursor::new(b"not an HDR file")).is_err());
        assert!(
            decode_hdr(Cursor::new(
                b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 4 +X 8\n"
            ))
            .is_err()
        );
        for dimensions in ["+Y 4 +X 8", "-Y 1 +X 8", "-Y 8192 +X 8192"] {
            let data = format!("#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n{dimensions}\n");
            assert!(decode_hdr(Cursor::new(data)).is_err());
        }
        for metadata in ["EXPOSURE=0", "EXPOSURE=NaN", "COLORCORR=1 -1 1"] {
            let data = format!("#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n{metadata}\n\n-Y 4 +X 8\n");
            assert!(decode_hdr(Cursor::new(data)).is_err());
        }
    }
}
