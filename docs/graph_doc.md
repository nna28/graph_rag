# Graph RAG

## Tạo vector db

```bash
python3 build_embed.py
```

## Lên server

```bash
python3 server.py
```

Có thể chỉnh độ dài đường đi đến các node xung quanh thực thể chính bằng cách chỉnh `d` tại:

```python 

#src/app.chat_endpoint

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    if rag_engine is None:
        raise HTTPException(status_code=500, detail="Hệ thống chưa khởi tạo xong")
    
    try:
    
        response_text = rag_engine.query(request.message, d=1)
        return {"response": response_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```



Lấy front-end.html để test.
