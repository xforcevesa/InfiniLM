use crate::Template;

#[inline]
pub(super) fn apply_chat(prompt: &str, template: Template) -> String {
    match template {
        Template::Chat9G => todo!(),
        Template::ChatTinyLlama => format!(
            "<|user|>\n{}</s>\n<|assistant|>\n",
            prompt.replace(' ', "▁")
        ),
    }
}
