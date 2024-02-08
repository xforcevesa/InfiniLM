use super::{utok, Tokenizer};
use memmap2::Mmap;
use patricia_tree::PatriciaMap;
use std::{fs::File, path::Path};

/// 一个基于朴素词表的分词器。
pub struct VocabTxt {
    /// 词表。
    words: Vec<String>,
    /// 词汇的前缀树。
    trie: PatriciaMap<utok>,
    /// 词汇的最大长度。
    max_piece_len: usize,
    /// 单字节词汇转义。
    byte_pieces: [u8; 256],
}

impl VocabTxt {
    pub fn new(tokenizer: impl AsRef<Path>) -> Self {
        let mmap = unsafe { Mmap::map(&File::open(tokenizer).unwrap()) }.unwrap();
        let text = unsafe { std::str::from_utf8_unchecked(&mmap) };

        let mut words = Vec::new();
        let mut trie = PatriciaMap::new();
        let mut max_piece_len = 0;
        for (i, line) in text.lines().into_iter().enumerate() {
            let piece = line.strip_prefix('"').unwrap().strip_suffix('"').unwrap();
            max_piece_len = max_piece_len.max(piece.len());
            words.push(piece.to_string());
            trie.insert(piece, i as _);
        }
        let mut ans = Self {
            words,
            trie,
            max_piece_len,
            byte_pieces: [0; 256],
        };
        for i in 0..=255u8 {
            ans.byte_pieces[i as usize] = i;
        }
        ans
    }
}

impl Tokenizer for VocabTxt {
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

    fn encode(&self, mut text: &str, bos: bool, eos: bool) -> Vec<utok> {
        let mut tokens = Vec::<utok>::new();
        if bos {
            tokens.push(self.bos());
        }

        while !text.is_empty() {
            let piece = if text.len() > self.max_piece_len {
                &text[..self.max_piece_len]
            } else {
                text
            };
            if let Some((pre, tok)) = self.trie.get_longest_common_prefix(piece) {
                tokens.push(*tok);
                text = &text[pre.len()..];
            } else {
                let mut chars = text.chars();
                let char = chars.next().unwrap();
                tokens.extend(char.to_string().bytes().map(|b| (b + 3) as utok));
                text = chars.as_str();
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

    fn decode(&self, token: utok, next: utok) -> &str {
        let piece = self.words[next as usize].as_str();
        if let Some(byte) = piece.strip_prefix("<0x").and_then(|s| s.strip_suffix('>')) {
            let byte = u8::from_str_radix(byte, 16).unwrap();
            let byte = &self.byte_pieces[byte as usize..][..1];
            unsafe { std::str::from_utf8_unchecked(byte) }
        } else if token == self.bos() && piece.starts_with(' ') {
            &piece[1..]
        } else {
            piece
        }
    }
}
