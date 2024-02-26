use crate::{ByteDecoder, Tokenizer};
use common::utok;
use std::{io::Result, path::Path};

/// 由 tokenizer.model 文件定义的 bpe 分词器。
///
/// 文件格式为 `[10, total_len, 10, str_len, [str;str_len], 21, [score;4], ..; vocab_size]`。
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
    /// 打开 tokenizer.model 文件并构造一个 bpe 分词器。
    pub fn from_model_file(model_file: impl AsRef<Path>) -> Result<Self> {
        // 打开文件
        let file = std::fs::File::open(model_file)?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }?;
        // 遍历文件，标记所有词汇的位置并记录最大长度
        let mut max_piece_len = 0usize;
        let offsets = (0..)
            .scan(0usize, |offset, _| match &mmap[*offset..] {
                [10, total_len, 10, str_len, ..] => {
                    max_piece_len = max_piece_len.max(*str_len as usize);
                    let next = *offset + 3;
                    *offset += 2 + *total_len as usize;
                    Some(next)
                }
                [..] => None,
            })
            .collect::<Vec<_>>();
        // 对词汇表按字典序排序
        let mut sorted_indices = (0..offsets.len() as utok).collect::<Vec<_>>();
        sorted_indices.sort_by_key(|&i| {
            let slice = &mmap[offsets[i as usize]..];
            let len = slice[0] as usize;
            std::str::from_utf8(&slice[1..][..len]).unwrap()
        });
        // 生成分词器
        Ok(Self {
            mmap,
            offsets,
            sorted_indices,
            max_piece_len: 0,
            byte_pieces: ByteDecoder::new(),
        })
    }

    /// 根据词汇查找代码。
    #[inline]
    fn find_piece(&self, piece: &str) -> Option<utok> {
        self.sorted_indices
            .binary_search_by_key(&piece, |&i| self.get_piece(i))
            .ok()
            .map(|i| self.sorted_indices[i])
    }

    /// 根据代码查找词汇。
    #[inline]
    fn get_piece(&self, i: utok) -> &str {
        let offset = self.offsets[i as usize];
        let slice = &self.mmap[offset..];
        let len = slice[0] as usize;
        std::str::from_utf8(&slice[1..][..len]).unwrap()
    }

    /// 根据代码查找合词评分。
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
    fn vocab_size(&self) -> usize {
        self.offsets.len()
    }

    #[inline]
    fn max_piece_len(&self) -> usize {
        self.max_piece_len
    }

    fn encode(&self, text: &str) -> Vec<utok> {
        let mut tokens = Vec::new();
        if let Some(c) = text.chars().next() {
            if c.is_alphabetic() {
                tokens.push(self.find_piece("▁").unwrap())
            }
        }

        text.chars().map(|c| c.to_string()).for_each(|c| {
            if let Some(index) = self.find_piece(&c) {
                tokens.extend([index]);
            } else {
                tokens.extend(c.bytes().map(|c| c as utok + 3));
            }
        });

        fn map_pair(bpe: &BPE, tokens: &[utok], i: usize) -> Option<(utok, f32)> {
            bpe.find_piece(&format!(
                "{}{}",
                bpe.get_piece(tokens[i]),
                bpe.get_piece(tokens[i + 1])
            ))
            .map(|tok| (tok, bpe.get_score(tok)))
        }

        let mut merges = (0..tokens.len() - 1)
            .map(|i| map_pair(self, &tokens, i))
            .collect::<Vec<_>>();
        while let Some((i, (tok, _))) = merges
            .iter()
            .enumerate()
            .filter_map(|(i, tok)| tok.map(|tok| (i, tok)))
            .max_by(|(_, (_, a)), (_, (_, b))| a.total_cmp(b))
        {
            tokens[i] = tok;
            tokens.remove(i + 1);
            merges.remove(i);
            if let Some(i) = i.checked_sub(1) {
                merges[i] = map_pair(self, &tokens, i);
            }
            if i + 1 < merges.len() {
                merges[i] = map_pair(self, &tokens, i);
            }
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
    if let Ok(bpe) = BPE::from_model_file("../../TinyLlama-1.1B-Chat-v1.0/tokenizer.model") {
        for i in 0..bpe.offsets.len() {
            println!("{}: {}", bpe.get_piece(i as utok), bpe.get_score(i as utok));
        }
    }
}

#[test]
fn once_upon_a_time() {
    use std::time::Instant;
    if let Ok(bpe) = BPE::from_model_file("../../TinyLlama-1.1B-Chat-v1.0/tokenizer.model") {
        const PROMPT: &str = "Once▁upon▁a▁time,";
        let tokens = bpe.encode(PROMPT);
        let t0 = Instant::now();
        for _ in 0..10000 {
            let _tokens = bpe.encode(PROMPT);
        }
        let t1 = Instant::now();
        println!("{:?}", t1 - t0);
        assert_eq!(tokens, &[9038, 2501, 263, 931, 29892]);
    }
}
