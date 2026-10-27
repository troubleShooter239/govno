#[derive(Default)]
pub(crate) struct Measurements {
    pub cpu: Vec<f64>,
    pub record: Vec<f64>,
    pub gpu: Vec<[f64; 4]>,
    pub enabled: bool,
    pub submitted: u64,
    pub culled: u64,
    pub shadow: u64,
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
    pub fn gpu_sample(&mut self, value: [f64; 4]) {
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
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn percentiles_use_nearest_rank() {
        let values: Vec<f64> = (1..=100).map(f64::from).collect();
        let d = distribution(&values);
        assert_eq!(d["p95_ms"], 95.0);
        assert_eq!(d["p99_ms"], 99.0);
        assert_eq!(d["mean_ms"], 50.5);
        assert!(distribution(&[]).is_null());
    }
}
