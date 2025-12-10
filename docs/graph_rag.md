# Graph RAG Multi-hop

Mô tả chi tiết pipeline `SmartGraphRAG` theo code hiện tại (`src/graph_rag.py`), gồm trích xuất thực thể, tìm neo, suy luận multi-hop, rerank và sinh câu trả lời.

## Luồng suy luận
- **Trích xuất thực thể**: prompt dạng System/User buộc chỉ trả về danh từ riêng xuất hiện trong câu hỏi (người/tổ chức/địa danh), không hội thoại, không đoán thêm; nếu nhiều thực thể thì cách nhau dấu phẩy.
- **Tìm node neo**: mỗi thực thể được tìm trong vector store `Chroma` (embedding `bkai-foundation-models/vietnamese-bi-encoder`). Lấy tối đa `anchor_per_entity`/thực thể và cắt tổng ở `max_anchors`, giữ thứ tự tìm thấy.
- **Lấy lân cận + rerank**: dựng `ego_graph` bán kính `depth` (alias `d`) quanh các neo, duyệt tối đa `neighbor_candidate_multiplier × neighbor_top_k` cạnh; mỗi cạnh được format rõ `Cạnh: A --rel--> B` để tránh bị gộp token. Rerank cạnh bằng cosine giữa embedding câu hỏi và embedding chuỗi cạnh, giữ `neighbor_top_k`.
- **Đường đi multi-hop + rerank**: nếu có ≥2 neo, tìm đường đi đơn giản trong đồ thị vô hướng tạm với `cutoff=max_hops`, duyệt tối đa `path_candidate_multiplier × top_k_paths` đường; mỗi đường hiển thị các cạnh tách bằng dấu chấm phẩy (`A -[rel]-> B ; B -[rel2]-> C`) để giảm nhầm lẫn token. Rerank các đường theo embedding câu hỏi, giữ `top_k_paths`.
- **Sinh câu trả lời**: ghép context (lân cận + multi-hop đã rerank) vào prompt. Nếu thiếu dữ kiện, model được yêu cầu trả về thông báo thiếu thay vì bịa.

## Cách chạy nhanh
1) Tạo vector cho các node (đọc `data/final/edges.csv`):  
`python build_embed.py`

2) Khởi động server (load LLM, nạp đồ thị từ `src/init_graph.py`, dùng vector đã persist trong `chroma/` nếu có):  
`python server.py`

3) Gọi suy luận trong code:
```python
from src.graph_rag import SmartGraphRAG, load_tiny_vietnamese_llm
from src.init_graph import init

llm = load_tiny_vietnamese_llm()
rag = SmartGraphRAG(llm_model=llm)
init(rag)  # nạp cạnh

answer = rag.query(
    "Aage Bohr có những giải thưởng gì?",
    depth=1,                 # bán kính lân cận
    anchor_per_entity=3,     # số neo / thực thể
    max_anchors=10,          # giới hạn tổng neo
    neighbor_top_k=4,        # số cạnh lân cận giữ lại sau rerank
    neighbor_candidate_multiplier=3,  # duyệt 3x rồi cắt xuống neighbor_top_k
    max_hops=3,              # chiều dài đường multi-hop
    top_k_paths=2,           # số đường multi-hop giữ lại sau rerank
    path_candidate_multiplier=3       # duyệt 3x rồi cắt xuống top_k_paths
)
print(answer)
```

## Tham số `query` quan trọng
- `depth` / `d`: bán kính ego-graph lấy cạnh lân cận (mặc định 1).
- `anchor_per_entity`, `max_anchors`: neo tối đa/ thực thể và tổng neo (mặc định 3, 10).
- `neighbor_top_k`, `neighbor_candidate_multiplier`: số cạnh lân cận giữ sau rerank và hệ số mở rộng trước khi cắt (4, 3).
- `max_hops`: số cạnh tối đa trên một đường multi-hop (3).
- `top_k_paths`, `path_candidate_multiplier`: số đường multi-hop giữ sau rerank và hệ số mở rộng (2, 3).

Nếu không tìm thấy liên kết hay đường đi phù hợp, hệ thống trả về thông báo thiếu dữ kiện để tránh bịa.***
