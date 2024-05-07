#![deny(warnings)]

mod sample;

#[derive(Clone, PartialEq, Debug)]
pub struct SampleArgs {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
}

impl Default for SampleArgs {
    #[inline]
    fn default() -> Self {
        Self {
            temperature: 0.,
            top_k: usize::MAX,
            top_p: 1.,
        }
    }
}
