use crate::{utok, Tokenizer};
use std::path::Path;

pub struct BPE {
    mmap: memmap2::Mmap,
    offsets: Vec<usize>,
}

impl BPE {
    pub fn from_model(model_file: impl AsRef<Path>) -> Self {
        let file = std::fs::File::open(model_file).unwrap();
        let mmap = unsafe { memmap2::Mmap::map(&file) }.unwrap();
        // format: 10 <total_len> 10 <str_len> <str;str_len> 21 <score;4> []
        let mut offsets = Vec::new();
        let mut offset = 0usize;
        loop {
            let slice = &mmap[offset..];
            if slice.is_empty() || slice[0] != 10 {
                break;
            }
            offsets.push(offset + 3);
            offset += 2 + slice[1] as usize;
        }
        Self { mmap, offsets }
    }

    #[inline]
    fn get_piece(&self, i: utok) -> &str {
        let offset = self.offsets[i as usize];
        let slice = &self.mmap[offset..];
        let len = slice[0] as usize;
        std::str::from_utf8(&slice[1..][..len]).unwrap()
    }

    #[inline]
    fn get_score(&self, i: utok) -> f32 {
        let offset = self.offsets[i as usize];
        let slice = &self.mmap[offset..];
        let len = slice[0] as usize;
        let ptr = slice[len + 2..].as_ptr().cast::<f32>();
        unsafe { ptr.read_unaligned() }
    }
}

impl Tokenizer for BPE {
    fn bos(&self) -> crate::utok {
        todo!()
    }

    fn eos(&self) -> crate::utok {
        todo!()
    }

    fn max_piece_len(&self) -> usize {
        todo!()
    }

    fn encode(&self, text: &str, bos: bool, eos: bool) -> Vec<crate::utok> {
        todo!()
    }

    fn decode(&self, token: crate::utok, next: crate::utok) -> &str {
        todo!()
    }
}

#[test]
fn read_tokenizer() {
    let bpe = BPE::from_model("tokenizer.model");
    for i in 0..bpe.offsets.len() {
        println!("{}: {}", bpe.get_piece(i as utok), bpe.get_score(i as utok));
    }
}
