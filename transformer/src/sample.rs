#[derive(Clone, PartialEq, Debug)]
pub enum SampleArgs {
    Top,
    Random {
        temperature: f32,
        top_k: usize,
        top_p: f32,
    },
}
