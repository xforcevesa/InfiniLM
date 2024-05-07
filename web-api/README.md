# web 服务

实现 web 服务，提供 RESTful API 接口。

```json
{
    "request": {
        "/infer": {
            "session_id": "String?",
            "inputs": [{
                "role": "String",
                "content": "String"
            }],
            "dialog_pos": "int?",
            "temperature": "float?",
            "top-k": "int?",
            "top-p": "float?",
        },
        "/fork": {
            "session_id": "String",
            "new_session_id": "String"
        },
        "/drop": {
            "session_id": "String"
        }
    },
    "response": {
        "session_not_found": {
            "status": 404,
            "code": 0,
            "message": "Session not found"
        },
        "session_busy": {
            "status": 406,
            "code": 0,
            "message": "Session is busy"
        },
        "session_duplicate": {
            "status": 409,
            "code": 0,
            "message": "Session ID already exists"
        },
        "empty_input": {
            "status": 400,
            "code": 0,
            "message": "Input list is empty"
        },
        "invalid_dialog_pos": {
            "status": 416,
            "code": 0,
            "message": "Dialog position out of range",
            "current_dialog_pos": "int"
        }
    }
}
```
