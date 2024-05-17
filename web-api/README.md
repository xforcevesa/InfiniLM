# web 服务

实现 web 服务，定义 RPC 风格的 web API。

## 目录

- [`POST /infer`](#post-infer)
- [`POST /fork`](#post-fork)
- [`POST /drop`](#post-drop)
- [错误类型](#错误类型)

## `POST /infer`

```json
"messages": [{
    "role": "user | assistant",
    "content": "string"
}],
"session_id": "string?",
"dialog_pos": "integer?=0",
"temperature": "number?",
"top-k": "integer?",
"top-p": "number?"
```

向 `session_id` 指定的会话或匿名会话的 `dialog_pos` 位置处连接 `messages`，并进行推理。

- `messages` 是必要的，但可以为空列表，不存在时返回[json 解析错误](#json-解析失败)；
- `dialog_pos` 不存在：视作 0；
- `dialog_pos` 为 0
  - `session_id` 不存在
    - `messages` 中最后一个消息 `role==user`：创建一个匿名会话并推理，匿名会话将在结束后立即清除；
    - `messages` 中最后一个消息 `role!=user`：返回一个立即结束的流；
  - `session_id` 存在
    - 会话存在
      - 会话状态忙：返回[会话忙错误](#会话忙)；
      - 会话状态空闲：将会话的状态清空，重置为 `messages`
        - `messages` 中最后一个消息 `role==user`：开始推理；
        - `messages` 中最后一个消息 `role!=user`：返回一个立即结束的流；
    - 会话不存在：创建一个新会话并填充 `messages`
      - `messages` 中最后一个消息 `role==user`：开始推理；
      - `messages` 中最后一个消息 `role!=user`：返回一个立即结束的流；
- `dialog_pos` 不为 0
  - `session_id` 不存在：返回[非法对话位置错误](#非法对话位置)；
  - `session_id` 存在
    - 会话存在
      - 会话状态忙：返回[会话忙错误](#会话忙)；
      - 会话状态空闲
        - 会话句子数不小于 `dialog_pos`：回滚到 `dialog_pos` 位置并连接 `messages`
          - `messages` 中最后一个消息 `role==user`：开始推理；
          - `messages` 中最后一个消息 `role!=user`：返回一个立即结束的流；
        - 会话句子数小于 `dialog_pos`：返回[非法对话位置错误](#非法对话位置)；
      - 会话不存在：返回[会话不存在错误](#会话不存在)；

## `POST /fork`

```json
"session_id": "string",
"new_session_id": "string"
```

将 `session_id` 指定的会话复制一份，并将新会话的 ID 设为 `new_session_id`。

- 会话不存在：返回[会话不存在错误](#会话不存在)；
- 会话存在
  - 会话状态忙：返回[会话忙错误](#会话忙)；
  - 会话状态空闲
    - `new_session_id` 已存在：返回[会话重复错误](#会话重复)；
    - `new_session_id` 不存在：复制会话；

## `POST /drop`

```json
"session_id": "string",
"new_session_id": "string"
```

删除 `session_id` 指定的会话。

- 会话不存在：返回[会话不存在错误](#会话不存在)；
- 会话存在：删除会话；

## 错误类型

### json 解析失败

```json
"status": 400,
"code": 0,
"message": "(Some json error)"
```

### 会话不存在

```json
"status": 404,
"code": 0,
"message": "Session not found"
```

### 会话忙

```json
"status": 406,
"code": 0,
"message": "Session is busy"
```

### 会话重复

```json
"status": 409,
"code": 0,
"message": "Session ID already exists"
```

### 非法对话位置

```json
"status": 416,
"code": 0,
"message": "Dialog position out of range",
"current_dialog_pos": "int"
```
