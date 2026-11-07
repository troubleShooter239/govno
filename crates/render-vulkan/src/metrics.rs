#[derive(Default)]
pub(crate) struct Measurements {
    pub cpu: Vec<f64>,
    pub record: Vec<f64>,
    pub gpu: Vec<[f64; 7]>,
    pub enabled: bool,
    pub submitted: u64,
    pub culled: u64,
    pub occluded: u64,
    pub shadow: u64,
    pub descriptor_binds: u64,
    pub indirect_calls: u64,
    pub depth_prepass_calls: u64,
}
impl Measurements {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn enable(&mut self) {
        self.enabled = true;
        self.cpu.reserve(4096);
        self.record.reserve(4096);
        self.gpu.reserve(4096);
    }
    pub fn gpu_sample(&mut self, value: [f64; 7]) {
        if self.enabled && self.gpu.len() < 100000 {
            self.gpu.push(value);
        }
    }
}
pub(crate) fn distribution(values: &[f64]) -> serde_json::Value {
    if values.is_empty() {
        return serde_json::Value::Null;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let quantile = |p: f64| {
        sorted[((p * sorted.len() as f64).ceil() as usize)
            .saturating_sub(1)
            .min(sorted.len() - 1)]
    };
    serde_json::json!({"samples":sorted.len(),"mean_ms":sorted.iter().sum::<f64>()/sorted.len()as f64,"p50_ms":quantile(0.5),"p95_ms":quantile(0.95),"p99_ms":quantile(0.99)})
}
