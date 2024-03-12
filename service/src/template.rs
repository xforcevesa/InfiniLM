use std::borrow::Cow;

pub trait Template {
    fn normalize<'a>(&self, prompt: &'a str) -> Cow<'a, str>;
    fn apply_chat<'a>(&self, prompt: &'a str) -> Cow<'a, str>;
    fn decode<'a>(&self, piece: &'a str) -> Cow<'a, str>;
}

pub struct ChatCPM;

pub struct ChatTinyLlama;

impl Template for ChatCPM {
    #[inline]
    fn normalize<'a>(&self, prompt: &'a str) -> Cow<'a, str> {
        Cow::Owned(format!("<s>{}", prompt.trim()))
    }

    #[inline]
    fn apply_chat<'a>(&self, prompt: &'a str) -> Cow<'a, str> {
        Cow::Owned(format!("<s><用户>{}<AI>", prompt.trim()))
    }

    #[inline]
    fn decode<'a>(&self, piece: &'a str) -> Cow<'a, str> {
        Cow::Borrowed(piece)
    }
}

impl Template for ChatTinyLlama {
    #[inline]
    fn normalize<'a>(&self, prompt: &'a str) -> Cow<'a, str> {
        let prompt = prompt.trim();

        let mut ans = String::new();
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
        Cow::Owned(ans)
    }

    #[inline]
    fn apply_chat<'a>(&self, prompt: &'a str) -> Cow<'a, str> {
        Cow::Owned(format!(
            "<|user|>\n{}</s><|assistant|>\n",
            self.normalize(prompt)
        ))
    }

    #[inline]
    fn decode<'a>(&self, piece: &'a str) -> Cow<'a, str> {
        Cow::Owned(piece.replace('▁', " "))
    }
}
