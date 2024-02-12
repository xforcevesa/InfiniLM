use crate::{utok, ByteDecoder, Tokenizer};
use std::{io::Result, path::Path};

pub struct BPE {
    mmap: memmap2::Mmap,
    /// 保存每个序号对应的对象在文件中的偏移，用于从序号查询 token 字符串。
    offsets: Vec<usize>,
    /// 保存根据 token 字符串字典序排序的序号，用于从 token 字符串查询序号。
    sorted_indices: Vec<utok>,
    max_piece_len: usize,
    byte_pieces: ByteDecoder,
}

impl BPE {
    pub fn from_model(model_file: impl AsRef<Path>) -> Result<Self> {
        let file = std::fs::File::open(model_file)?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }?;
        // format: 10 <total_len> 10 <str_len> <str;str_len> 21 <score;4> []
        let mut offsets = Vec::new();
        let mut offset = 0usize;
        let mut max_piece_len = 0usize;
        loop {
            let slice = &mmap[offset..];
            if slice.is_empty() || slice[0] != 10 {
                break;
            }
            max_piece_len = max_piece_len.max(slice[3] as usize);
            offsets.push(offset + 3);
            offset += 2 + slice[1] as usize;
        }
        let mut sorted_indices = (0..offsets.len() as utok).collect::<Vec<_>>();
        sorted_indices.sort_by_key(|&i| {
            let slice = &mmap[offsets[i as usize]..];
            let len = slice[0] as usize;
            std::str::from_utf8(&slice[1..][..len]).unwrap()
        });
        Ok(Self {
            mmap,
            offsets,
            sorted_indices,
            max_piece_len: 0,
            byte_pieces: ByteDecoder::new(),
        })
    }

    #[inline]
    fn find_piece(&self, piece: &str) -> Option<utok> {
        self.sorted_indices
            .binary_search_by_key(&piece, |&i| self.get_piece(i))
            .ok()
            .map(|i| self.sorted_indices[i])
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
    #[inline]
    fn bos(&self) -> utok {
        1
    }

    #[inline]
    fn eos(&self) -> utok {
        2
    }

    #[inline]
    fn max_piece_len(&self) -> usize {
        self.max_piece_len
    }

    fn encode(&self, text: &str, bos: bool, eos: bool) -> Vec<utok> {
        let mut tokens = Vec::new();
        if bos {
            tokens.push(self.bos());
        }
        if !text.is_empty() {
            tokens.push(self.find_piece("▁").unwrap())
        }

        text.chars().map(|c| c.to_string()).for_each(|c| {
            if let Some(index) = self.find_piece(&c) {
                tokens.extend([index]);
            } else {
                tokens.extend(c.bytes().map(|c| c as utok + 3));
            }
        });

        fn map_pair(bpe: &BPE, tokens: &[utok], i: usize) -> Option<utok> {
            bpe.find_piece(&format!(
                "{}{}",
                bpe.get_piece(tokens[i]),
                bpe.get_piece(tokens[i + 1])
            ))
        }

        let mut merges = (0..tokens.len() - 1)
            .map(|i| map_pair(self, &tokens, i))
            .collect::<Vec<_>>();
        loop {
            let mut best_score = std::f32::NEG_INFINITY;
            let mut replacement = None;
            for (i, index) in merges.iter().enumerate() {
                if let Some(index) = index {
                    let score = self.get_score(*index);
                    if score > best_score {
                        best_score = score;
                        replacement = Some((i, *index));
                    }
                }
            }
            match replacement {
                Some((i, j)) => {
                    tokens[i] = j;
                    tokens.remove(i + 1);
                    merges.remove(i);
                    if let Some(index) = merges.get_mut(i - 1) {
                        *index = map_pair(self, &tokens, i - 1);
                    }
                    if let Some(index) = merges.get_mut(i) {
                        *index = map_pair(self, &tokens, i);
                    }
                }
                None => break,
            }
        }

        if bos {
            assert_eq!(tokens[0], self.bos());
        }
        if eos {
            tokens.push(self.eos());
        }
        tokens
    }

    #[inline]
    fn decode(&self, token: utok) -> &str {
        self.byte_pieces.decode(self.get_piece(token))
    }
}

#[test]
fn read_tokenizer() {
    if let Ok(bpe) = BPE::from_model("tokenizer.model") {
        for i in 0..bpe.offsets.len() {
            println!("{}: {}", bpe.get_piece(i as utok), bpe.get_score(i as utok));
        }
    }
}

#[test]
fn once_upon_a_time() {
    use std::time::Instant;
    if let Ok(bpe) = BPE::from_model("tokenizer.model") {
        let tokens = bpe.encode("Once▁upon▁a▁time,", true, false);
        let t0 = Instant::now();
        for _ in 0..10000 {
            let _tokens = bpe.encode("Once▁upon▁a▁time,", true, false);
        }
        let t1 = Instant::now();
        println!("{:?}", t1 - t0);
        assert_eq!(tokens, &[1, 9038, 2501, 263, 931, 29892]);
    }
}
