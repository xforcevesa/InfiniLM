#![deny(warnings)]

mod sample;

/// 采样参数。
#[derive(Clone, PartialEq, Debug)]
pub struct SampleArgs {
    /// 温度，大于 0 有效，否则使用贪心采样。
    pub temperature: f32,
    /// 硬阈值，大于 1 有效，否则使用贪心采样。
    pub top_k: usize,
    /// 软阈值，(0, 1] 区间有效，不大于 0 使用贪心采样。
    pub top_p: f32,
}

impl Default for SampleArgs {
    #[inline]
    fn default() -> Self {
        // 默认值通过指示温度为 0 使用贪心采样
        Self {
            temperature: 0.,
            top_k: usize::MAX,
            top_p: 1.,
        }
    }
}
