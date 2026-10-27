use crate::{HdrPanorama, MAX_HDR_PIXELS};
use glam::Vec3;
use govno_core::BoxError;
use image::{ImageDecoder, codecs::openexr::OpenExrDecoder};
use std::{
    fs::File,
    io::{BufReader, Read, Seek, SeekFrom},
    path::Path,
};
pub fn load_environment(path: impl AsRef<Path>) -> Result<HdrPanorama, BoxError> {
    let path = path.as_ref();
    let result = (|| {
        let mut reader = BufReader::new(File::open(path)?);
        let mut magic = [0; 4];
        reader.read_exact(&mut magic)?;
        reader.seek(SeekFrom::Start(0))?;
        if magic == [0x76, 0x2f, 0x31, 0x01] {
            decode_exr(reader)
        } else {
            crate::decode_hdr(reader)
        }
    })();
    result.map_err(|e| format!("environment {}: {e}", path.display()).into())
}
pub fn decode_exr(reader: impl Read + Seek) -> Result<HdrPanorama, BoxError> {
    let decoder = OpenExrDecoder::with_alpha_preference(BufReader::new(reader), Some(false))?;
    let (width, height) = decoder.dimensions();
    if width < 4 || height < 2 || u64::from(width) * u64::from(height) > MAX_HDR_PIXELS {
        return Err("EXR panorama dimensions exceed supported limits".into());
    }
    let image = image::DynamicImage::from_decoder(decoder)?.into_rgb32f();
    let radiance: Vec<_> = image.pixels().map(|p| Vec3::from_array(p.0)).collect();
    if radiance
        .iter()
        .any(|v| !v.is_finite() || v.min_element() < 0.0)
    {
        return Err("EXR requires finite nonnegative linear RGB".into());
    }
    Ok(HdrPanorama {
        width,
        height,
        radiance,
        exposure: None,
        color_correction: None,
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    use image::{ExtendedColorType, ImageEncoder, codecs::openexr::OpenExrEncoder};
    use std::io::Cursor;
    fn encode(value: f32) -> Vec<u8> {
        let mut file = Cursor::new(Vec::new());
        let bytes: Vec<u8> = (0..8)
            .flat_map(|_| [value, 0.5, 2.0])
            .flat_map(f32::to_ne_bytes)
            .collect();
        OpenExrEncoder::new(&mut file)
            .write_image(&bytes, 4, 2, ExtendedColorType::Rgb32F)
            .unwrap();
        file.into_inner()
    }
    #[test]
    fn exr_float_energy_and_order_survive_decode() {
        let map = decode_exr(Cursor::new(encode(4.25))).unwrap();
        assert_eq!((map.width, map.height), (4, 2));
        assert_eq!(map.radiance[0], Vec3::new(4.25, 0.5, 2.0));
        assert!(decode_exr(Cursor::new(encode(-1.0))).is_err());
        assert!(decode_exr(Cursor::new(b"bad EXR")).is_err());
    }
}
