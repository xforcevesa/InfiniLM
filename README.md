# 手写 transformer 模型

从 [YdrMaster/llama2.rs](https://github.com/YdrMaster/llama2.rs) 发展来的手写 transformer 模型项目。

## 使用

> 推荐测试模型：[TinyLlama-1.1B-Chat](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)。

> 下文所述“模型目录”，需要至少包含下列 3 个文件：
>
> - `config.json`: 模型配置文件；
> - `model.safetesnors`: 模型参数文件；
> - `tokenizer.model`/`vocab.txt`: 分词器词表；

### 转换参数

```plaintext
cargo cast --model <model> --dt <date_type>
```

用于转换参数类型以加速模型加载。

参数：

- `model`: 模型目录；

  生成的模型会存放在 `model` 同级目录下，并添加 `_<date_type>` 后缀。

- `date_type`: 参数类型，可为 `f32`/`f16`/`bf16`；

### 启动对话服务

```plaintext
cargo chat --model <model>
```

必要参数：

- `model`: 模型目录；

  > 目前仅支持 `f16` 精度，必须先转换模型；

其他参数参见 `cargo chat --help`。

### 启动文本生成

```plaintext
cargo generate --model <model> --prompt <prompt>
```

必要参数：

- `model`: 模型目录；

  > 目前仅支持 `f16` 精度，必须先转换模型。

- `prompt`: 生成文本的开头；

其他参数参见 `cargo generate --help`。


