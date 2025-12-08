
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Import module của ông
from src.graph_rag import SmartGraphRAG
from src.init_graph import init
from src.graph_rag import load_tiny_vietnamese_llm

load_dotenv()

app = FastAPI(title="Mark-2 GraphRAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


rag_engine = None

@app.on_event("startup")
async def startup_event():
    global rag_engine
    print("Đang khởi động hệ thống GraphRAG...")
    
    try:
        llm = load_tiny_vietnamese_llm()
        
        rag_engine = SmartGraphRAG(llm_model=llm)
        
        init(rag_engine)
        
        print("Hệ thống đã sẵn sàng!")
    except Exception as e:
        print(f"Lỗi khởi tạo: {e}")


class ChatRequest(BaseModel):
    message: str

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    if rag_engine is None:
        raise HTTPException(status_code=500, detail="Hệ thống chưa khởi tạo xong")
    
    try:
    
        response_text = rag_engine.query(request.message)
        return {"response": response_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "active", "model": "GraphRAG Mark-2"}
