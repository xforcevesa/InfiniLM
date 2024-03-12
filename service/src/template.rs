use std::borrow::Cow;

pub trait Template {
    fn encode<'a>(&self, prompt: &'a str) -> Cow<'a, str>;
    fn decode<'a>(&self, piece: &'a str) -> Cow<'a, str>;
}

pub struct ChatCPM;

pub struct ChatTinyLlama;

impl Template for ChatCPM {
    #[inline]
    fn encode<'a>(&self, prompt: &'a str) -> Cow<'a, str> {
        Cow::Owned(format!("<s><用户>{}<AI>", prompt.trim()))
    }

    #[inline]
    fn decode<'a>(&self, piece: &'a str) -> Cow<'a, str> {
        Cow::Borrowed(piece)
    }
}

impl Template for ChatTinyLlama {
    fn encode<'a>(&self, prompt: &'a str) -> Cow<'a, str> {
        let mut ans = String::from("<|user|>\n");
        match prompt.chars().next() {
            Some(c) if c.is_ascii_alphabetic() => ans.push('▁'),
            _ => {}
        }
        for c in prompt.chars() {
            ans.push(match c {
                ' ' => '▁',
                c => c,
            });
        }
        ans.push_str("</s><|assistant|>\n");
        Cow::Owned(ans)
    }

    #[inline]
    fn decode<'a>(&self, piece: &'a str) -> Cow<'a, str> {
        Cow::Owned(piece.replace('▁', " "))
    }
}
