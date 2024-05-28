//! See tokenizer_config.json/chat_template.

use std::borrow::Cow;

pub trait Template {
    fn normalize<'a>(&self, prompt: &'a str) -> Cow<'a, str>;
    fn apply_chat<'a>(&self, prompt: &'a str) -> Cow<'a, str>;
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
}

impl Template for ChatTinyLlama {
    #[inline]
    fn normalize<'a>(&self, prompt: &'a str) -> Cow<'a, str> {
        Cow::Borrowed(prompt)
    }

    #[inline]
    fn apply_chat<'a>(&self, prompt: &'a str) -> Cow<'a, str> {
        Cow::Owned(format!("<|user|>\n{prompt}</s><|assistant|>\n"))
    }
}
