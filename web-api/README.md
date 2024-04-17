# web 服务

实现 web 服务，提供 RESTful API 接口。

```json
{
    "/infer": {
        "session_id": "String",
        "inputs": [{
            "role": "String",
            "content": "String"
        }],
        "dialog_pos": "int"
    },
    "/fork": {
        "session_id": "String",
        "new_session_id": "String"
    },
    "/drop": {
        "session_id": "String"
    }
}
```
