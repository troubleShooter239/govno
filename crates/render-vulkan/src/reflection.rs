//! Slang reflection is generated with each SPIR-V file, fingerprinted, embedded
//! and validated against host descriptor/push/vertex/constant-buffer contracts.
use ash::vk;
use govno_core::BoxError;
use serde_json::Value;
use std::ffi::CStr;
fn document(entry: &str) -> Result<&'static str, BoxError> {
    macro_rules! files{($($name:literal),*)=>{match entry{$($name=>Ok(include_str!(concat!("../shaders/",$name,".reflection.json"))),)*_=>Err(format!("no shader reflection for {entry}").into())}}}
    files!(
        "vertexMain",
        "fragmentMain",
        "postVertex",
        "postFragment",
        "shadowVertex",
        "meterExposure",
        "bloomExtract",
        "bloomBlur",
        "prefilterIBL",
        "integrateBRDF",
        "projectSH"
    )
}
fn number(v: &Value, key: &str) -> Result<u32, BoxError> {
    Ok(u32::try_from(
        v[key]
            .as_u64()
            .ok_or_else(|| format!("reflection missing {key}"))?,
    )?)
}
fn uniform_size(v: &Value) -> Result<u32, BoxError> {
    let sizes = v["sizes"]
        .as_array()
        .ok_or("reflection missing uniform sizes")?;
    let value = sizes
        .iter()
        .find(|s| s["kind"] == "uniform")
        .ok_or("reflection missing uniform size")?;
    number(value, "value")
}
fn descriptor(v: &Value) -> Result<vk::DescriptorType, BoxError> {
    let t = &v["type"];
    Ok(match t["kind"].as_str() {
        Some("constantBuffer") => vk::DescriptorType::UNIFORM_BUFFER,
        Some("samplerState") => vk::DescriptorType::SAMPLER,
        Some("resource") => match t["baseShape"].as_str() {
            Some("structuredBuffer") => vk::DescriptorType::STORAGE_BUFFER,
            Some("texture2D") if t["access"] == "readWrite" => vk::DescriptorType::STORAGE_IMAGE,
            Some("texture2D") => vk::DescriptorType::SAMPLED_IMAGE,
            _ => return Err("unsupported reflected resource shape".into()),
        },
        _ => return Err("unsupported reflected descriptor type".into()),
    })
}
#[derive(Debug)]
pub(crate) struct Interface {
    pub bindings: Vec<(u32, u32, vk::DescriptorType)>,
    pub push_bytes: u32,
    root: Value,
}
fn parse(code: &[u8], entry: &str, json: &str) -> Result<Interface, BoxError> {
    let root: Value = serde_json::from_str(json)?;
    let hash = code.iter().fold(2166136261_u32, |h, b| {
        (h ^ u32::from(*b)).wrapping_mul(16777619)
    });
    if root["spirvFnv1a32"] != format!("{hash:08x}") {
        return Err(format!("SPIR-V/reflection fingerprint mismatch: {entry}").into());
    }
    if root["entryPoints"][0]["name"] != entry {
        return Err("reflection entry point mismatch".into());
    }
    let mut bindings = vec![];
    let mut push_bytes = 0;
    for p in root["parameters"]
        .as_array()
        .ok_or("reflection missing parameters")?
    {
        match p["binding"]["kind"].as_str() {
            Some("pushConstantBuffer") => push_bytes = uniform_size(&p["type"]["elementType"])?,
            Some("descriptorTableSlot") => bindings.push((
                p["binding"]["space"].as_u64().unwrap_or(0) as u32,
                number(&p["binding"], "index")?,
                descriptor(p)?,
            )),
            _ => return Err("unsupported reflection parameter binding".into()),
        }
    }
    bindings.sort_by_key(|b| (b.0, b.1));
    Ok(Interface {
        bindings,
        push_bytes,
        root,
    })
}
pub(crate) fn reflect(code: &[u8], entry: &CStr) -> Result<Interface, BoxError> {
    let name = entry.to_str()?;
    parse(code, name, document(name)?)
}
pub(crate) fn compute(
    code: &[u8],
    entry: &CStr,
    bindings: &[(u32, vk::DescriptorType)],
    push: u32,
) -> Result<Vec<(u32, vk::DescriptorType)>, BoxError> {
    let r = reflect(code, entry)?;
    let mut expected: Vec<_> = bindings.iter().map(|&(b, t)| (0, b, t)).collect();
    expected.sort_by_key(|b| (b.0, b.1));
    if r.bindings != expected || r.push_bytes != push {
        return Err(format!("compute reflection contract mismatch: {entry:?}").into());
    }
    for parameter in r.root["parameters"]
        .as_array()
        .ok_or("missing compute parameters")?
    {
        if parameter["binding"]["kind"] == "descriptorTableSlot" {
            let name = parameter["name"]
                .as_str()
                .ok_or("missing compute resource name")?;
            let expected = match name {
                "source" | "panorama" => 0,
                "exposureState" | "panoramaSampler" => 1,
                "target" | "atlas" => 2,
                "lut" => 3,
                "sh" => 4,
                _ => return Err("unknown compute resource role".into()),
            };
            if number(&parameter["binding"], "index")? != expected {
                return Err("reflected compute resource binding changed".into());
            }
        }
        if parameter["binding"]["kind"] == "pushConstantBuffer" {
            if push == 48 {
                fields(parameter, &[("exposure", 0), ("bloom", 16), ("timing", 32)])?;
            } else if push == 16 {
                fields(
                    parameter,
                    &[("probe", 0), ("samples", 4), ("pad0", 8), ("pad1", 12)],
                )?;
            }
        }
    }
    Ok(r.bindings.into_iter().map(|(_, b, t)| (b, t)).collect())
}
fn fields(parameter: &Value, expected: &[(&str, u32)]) -> Result<(), BoxError> {
    let fields = parameter["type"]["elementType"]["fields"]
        .as_array()
        .ok_or("missing reflected constant fields")?;
    for &(name, offset) in expected {
        let field = fields
            .iter()
            .find(|v| v["name"] == name)
            .ok_or_else(|| format!("missing reflected field {name}"))?;
        if number(&field["binding"], "offset")? != offset {
            return Err(format!("reflected constant offset mismatch: {name}").into());
        }
    }
    Ok(())
}
pub(crate) fn graphics(code: &[u8], entry: &CStr) -> Result<(), BoxError> {
    let r = reflect(code, entry)?;
    let entry = entry.to_str()?;
    let post = entry.starts_with("post");
    let shadow = entry == "shadowVertex";
    let push = if post {
        crate::post::OUTPUT_BYTES
    } else if shadow {
        64
    } else {
        crate::scene_gpu::DRAW_BYTES
    };
    if r.push_bytes != push {
        return Err(format!("push constant reflection mismatch: {entry}").into());
    }
    for &(set, binding, ty) in &r.bindings {
        let expected = if post {
            if set != 0 {
                return Err("post descriptor set mismatch".into());
            }
            match binding {
                0 | 1 => vk::DescriptorType::SAMPLED_IMAGE,
                2 => vk::DescriptorType::STORAGE_BUFFER,
                _ => return Err("post binding mismatch".into()),
            }
        } else {
            if set >= 4 || shadow {
                return Err("graphics descriptor set mismatch".into());
            }
            match binding {
                0 => vk::DescriptorType::UNIFORM_BUFFER,
                1 | 3 | 5 | 6 | 8 => vk::DescriptorType::SAMPLED_IMAGE,
                2 | 4 | 7 => vk::DescriptorType::SAMPLER,
                9 => vk::DescriptorType::STORAGE_BUFFER,
                _ => return Err("graphics binding mismatch".into()),
            }
        };
        if ty != expected {
            return Err(format!("descriptor reflection mismatch at {set}:{binding}").into());
        }
    }
    for parameter in r.root["parameters"]
        .as_array()
        .ok_or("missing shader parameters")?
    {
        if parameter["name"] == "draw" {
            fields(
                parameter,
                &[
                    ("model", 0),
                    ("inverseScaleSquaredRoughness", 64),
                    ("baseColorMetallic", 80),
                    ("normalParams", 96),
                    ("emissiveOcclusion", 112),
                ],
            )?;
        }
        if parameter["name"] == "settings" {
            fields(parameter, &[("manualSRGB", 0), ("bloomIntensity", 4)])?;
        }
        if parameter["binding"]["kind"] == "descriptorTableSlot" {
            let name = parameter["name"]
                .as_str()
                .ok_or("missing reflected resource name")?;
            let expected = match name {
                "frame" => (0, 0),
                "environmentSH" => (0, 9),
                "specularAtlas" => (0, 6),
                "environmentSampler" => (0, 7),
                "brdfLut" => (0, 8),
                "shadowMap" => (0, 5),
                "baseColorTexture" => (0, 1),
                "baseColorSampler" => (0, 2),
                "normalTexture" => (1, 1),
                "normalSampler" => (1, 2),
                "metalRoughTexture" => (2, 1),
                "metalRoughSampler" => (2, 2),
                "emissiveTexture" => (3, 1),
                "emissiveSampler" => (3, 2),
                "occlusionTexture" => (3, 3),
                "occlusionSampler" => (3, 4),
                "hdrImage" => (0, 0),
                "bloomImage" => (0, 1),
                "exposureState" => (0, 2),
                _ => return Err("unrecognized reflected graphics resource role".into()),
            };
            if (
                parameter["binding"]["space"].as_u64().unwrap_or(0) as u32,
                number(&parameter["binding"], "index")?,
            ) != expected
            {
                return Err(format!("reflected graphics resource binding changed: {name}").into());
            }
        }
        if parameter["name"] == "frame" {
            let t = &parameter["type"]["elementType"];
            if uniform_size(t)? as u64 != crate::scene_gpu::CAMERA_BYTES {
                return Err("reflected frame UBO size mismatch".into());
            }
            for (name, offset, stride) in [
                ("viewProjection", 0, 0),
                ("cameraPosition", 64, 0),
                ("lightDirection", 80, 0),
                ("lightRadiance", 96, 0),
                ("pointLights", 112, 32),
                ("shadowViewProjection", 624, 0),
                ("shadowParams", 688, 0),
                ("diffuseSH", 704, 16),
                ("environmentParams", 848, 0),
                ("cascadeMatrices", 864, 64),
                ("cascadeSplits", 1120, 0),
                ("cascadeParams", 1136, 0),
                ("localProbes", 1152, 48),
            ] {
                let field = t["fields"]
                    .as_array()
                    .ok_or("missing reflected fields")?
                    .iter()
                    .find(|f| f["name"] == name)
                    .ok_or_else(|| format!("missing UBO field {name}"))?;
                if number(&field["binding"], "offset")? != offset
                    || (stride > 0 && number(&field["binding"], "elementStride")? != stride)
                {
                    return Err(format!("reflected UBO offset/stride mismatch: {name}").into());
                }
            }
        }
    }
    if entry == "vertexMain" || shadow {
        let parameters = r.root["entryPoints"][0]["parameters"]
            .as_array()
            .ok_or("missing vertex parameters")?;
        let fields = if shadow {
            parameters.clone()
        } else {
            parameters[0]["type"]["fields"]
                .as_array()
                .ok_or("missing vertex input fields")?
                .clone()
        };
        let mut actual = vec![];
        for f in &fields {
            if f["binding"]["kind"] == "varyingInput" {
                actual.push((
                    number(&f["binding"], "index")?,
                    number(&f["type"], "elementCount")?,
                ));
            }
        }
        actual.sort();
        let expected = if shadow {
            vec![(0, 3)]
        } else {
            vec![(0, 3), (1, 3), (2, 2), (3, 4)]
        };
        if actual != expected {
            return Err("vertex attribute reflection mismatch".into());
        }
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn embedded_compute_interfaces_match_all_shared_layouts() {
        let effects = [
            (0, vk::DescriptorType::SAMPLED_IMAGE),
            (1, vk::DescriptorType::STORAGE_BUFFER),
            (2, vk::DescriptorType::STORAGE_IMAGE),
        ];
        let ibl = [
            (0, vk::DescriptorType::SAMPLED_IMAGE),
            (1, vk::DescriptorType::SAMPLER),
            (2, vk::DescriptorType::STORAGE_IMAGE),
            (3, vk::DescriptorType::STORAGE_IMAGE),
            (4, vk::DescriptorType::STORAGE_BUFFER),
        ];
        for (code, entry) in [
            (
                include_bytes!("../shaders/meterExposure.spv").as_slice(),
                c"meterExposure",
            ),
            (
                include_bytes!("../shaders/bloomExtract.spv").as_slice(),
                c"bloomExtract",
            ),
            (
                include_bytes!("../shaders/bloomBlur.spv").as_slice(),
                c"bloomBlur",
            ),
        ] {
            compute(code, entry, &effects, 48).unwrap();
        }
        for (code, entry) in [
            (
                include_bytes!("../shaders/prefilterIBL.spv").as_slice(),
                c"prefilterIBL",
            ),
            (
                include_bytes!("../shaders/integrateBRDF.spv").as_slice(),
                c"integrateBRDF",
            ),
            (
                include_bytes!("../shaders/projectSH.spv").as_slice(),
                c"projectSH",
            ),
        ] {
            compute(code, entry, &ibl, 16).unwrap();
        }
    }
    #[test]
    fn reflection_rejects_same_size_constant_field_reordering() {
        let mut root: Value = serde_json::from_str(document("fragmentMain").unwrap()).unwrap();
        let parameter = root["parameters"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|v| v["name"] == "draw")
            .unwrap();
        parameter["type"]["elementType"]["fields"][0]["binding"]["offset"] = serde_json::json!(16);
        assert!(fields(parameter, &[("model", 0)]).is_err());
    }
    #[test]
    fn all_embedded_graphics_interfaces_match_host_layouts() {
        for (code, entry) in [
            (
                include_bytes!("../shaders/vertexMain.spv").as_slice(),
                c"vertexMain",
            ),
            (
                include_bytes!("../shaders/fragmentMain.spv").as_slice(),
                c"fragmentMain",
            ),
            (
                include_bytes!("../shaders/postVertex.spv").as_slice(),
                c"postVertex",
            ),
            (
                include_bytes!("../shaders/postFragment.spv").as_slice(),
                c"postFragment",
            ),
            (
                include_bytes!("../shaders/shadowVertex.spv").as_slice(),
                c"shadowVertex",
            ),
        ] {
            graphics(code, entry).unwrap();
        }
    }
    #[test]
    fn stale_binary_and_mutated_push_layout_are_rejected() {
        let code = include_bytes!("../shaders/postFragment.spv");
        assert!(parse(&[0; 16], "postFragment", document("postFragment").unwrap()).is_err());
        assert!(compute(code, c"postFragment", &[], 128).is_err());
    }
}
