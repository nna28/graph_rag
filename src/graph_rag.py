import networkx as nx
import torch
import re
from typing import List, Optional, Set
import numpy as np
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from dotenv import load_dotenv

load_dotenv()

def load_tiny_vietnamese_llm():
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,  
        device_map="auto",
        trust_remote_code=True
    )

    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256, 
        
        
        do_sample=False,       
        repetition_penalty=1.05, 
        
        
        return_full_text=False
    )

    return HuggingFacePipeline(pipeline=pipe)

class SmartGraphRAG:
    def __init__(self, llm_model):
        self.llm = llm_model
        self.graph = nx.DiGraph()
        
        self.embedding_model = HuggingFaceEmbeddings(model_name="bkai-foundation-models/vietnamese-bi-encoder")
        
        self.vector_store = Chroma(
            collection_name="graph_nodes",
            embedding_function=self.embedding_model,
            persist_directory="chroma"
        )
        self.visited_nodes = set() 

        self.entity_extract_prompt = PromptTemplate(
            template="""System: Bạn là bộ trích xuất thực thể. 
            - Chỉ lấy danh từ riêng (người / tổ chức / địa danh) đã có trong câu hỏi.
            - Không đoán thêm thực thể mới, không trả lời câu hỏi, không hội thoại.
            - Chỉ trả về danh sách thực thể (chuỗi), phân tách bằng dấu phẩy nếu >1. Nếu không có, trả 'Không có'.
            
            Ví dụ đúng:
            Q: Quốc gia nào mà Alexander Prokhorov từng có quốc tịch?
            A: Alexander Prokhorov
            Q: Ai là người sáng lập Microsoft?
            A: Microsoft
            Q: Tập đoàn FPT thành lập năm nào?
            A: FPT
            Q: Sơn Tùng MTP quê ở đâu?
            A: Sơn Tùng MTP

            User: {question}
            Thực thể:""",
            input_variables=["question"]
        )
        self.entity_chain = self.entity_extract_prompt | self.llm | StrOutputParser()

    def add_triplet(self, subj: str, obj: str, rel: str):
        self.graph.add_edge(subj, obj, relation=rel)

    def build_vector(self, subj: str, obj: str, rel: str):
        if subj not in self.visited_nodes:
            self.vector_store.add_documents([Document(page_content=subj, metadata={"type": "node"})])
            self.visited_nodes.add(subj)
        if obj not in self.visited_nodes:
            self.vector_store.add_documents([Document(page_content=obj, metadata={"type": "node"})])
            self.visited_nodes.add(obj)

    def _clean_entities(self, raw_text: str):
        first_line = raw_text.strip().split('\n')[0]
        clean_text = re.sub(r'[.,;!?]+$', '', first_line)
        entities = [e.strip() for e in clean_text.split(',') if e.strip()]
        return entities

    def _search_anchor_nodes(
        self,
        target_entities: List[str],
        per_entity_k: int = 3,
        max_anchors: int = 10
    ) -> List[str]:
        anchors: List[str] = []
        for entity in target_entities:
            results = self.vector_store.similarity_search_with_score(entity, k=per_entity_k)
            for doc, _score in results:
                if doc.page_content not in anchors:
                    anchors.append(doc.page_content)
                if len(anchors) >= max_anchors:
                    return anchors
        return anchors

    def _format_edge(self, u: str, v: str, rel: str) -> str:
        # Thêm mô tả rõ ràng để giảm việc mô hình coi chuỗi là token liền nhau
        return f"{u} có {rel} là {v}."

    def _collect_neighbor_triplets(self, nodes: List[str], depth: int, max_edges: Optional[int] = None) -> List[str]:
        context_triplets: List[str] = []
        seen = set()
        for node in nodes:
            if not self.graph.has_node(node):
                continue
            subgraph = nx.ego_graph(self.graph, node, radius=depth)
            for u, v, data in subgraph.edges(data=True):
                rel = data.get('relation', 'liên quan')
                edge_text = self._format_edge(u, v, rel)
                if edge_text in seen:
                    continue
                seen.add(edge_text)
                context_triplets.append(edge_text)
                # if max_edges is not None and len(context_triplets) >= max_edges:
                #     return context_triplets
        return context_triplets

    def _edge_text(self, u: str, v: str):
        
        if self.graph.has_edge(u, v):
            rel = self.graph[u][v].get('relation', 'liên quan')
            return f"{u} có quan hệ '{rel}' với {v}."
            
        if self.graph.has_edge(v, u):
            rel = self.graph[v][u].get('relation', 'liên quan')
            return f"{v} có quan hệ '{rel}' với {u}."
            
        return f"{u} và {v} có liên quan."

    def _format_path(self, nodes: List[str]) -> str:
        parts = []
        for u, v in zip(nodes, nodes[1:]):
            parts.append(self._edge_text(u, v))
        # Dùng dấu chấm phẩy để tách cạnh, tránh bị gộp thành một token liền nhau
        return " ; ".join(parts)

    def _find_multi_hop_paths(
        self,
        anchors: List[str],
        max_hops: int = 3,
        candidate_limit: int = 20
    ) -> List[str]:
        if len(anchors) < 2:
            return []

        undirected_graph = self.graph.to_undirected()
        paths: List[str] = []
        anchor_list = list(anchors)

        for i in range(len(anchor_list)):
            for j in range(i + 1, len(anchor_list)):
                src = anchor_list[i]
                dst = anchor_list[j]

                if not undirected_graph.has_node(src) or not undirected_graph.has_node(dst):
                    continue

                for path in nx.all_simple_paths(undirected_graph, source=src, target=dst, cutoff=max_hops):
                    if len(path) < 2:
                        continue
                    formatted_path = self._format_path(path)
                    if formatted_path:
                        paths.append(formatted_path)
                    if len(paths) >= candidate_limit:
                        return paths

        return paths

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        va = np.array(a)
        vb = np.array(b)
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        if denom == 0:
            return 0.0
        return float(np.dot(va, vb) / denom)

    def _rerank_texts(self, texts: List[str], query: str, top_k: int) -> List[str]:
        if not texts:
            return []
        if top_k <= 0:
            return []
        query_vec = self.embedding_model.embed_query(query)
        doc_vecs = self.embedding_model.embed_documents(texts)
        scored = []
        for text, vec in zip(texts, doc_vecs):
            scored.append((text, self._cosine_similarity(query_vec, vec)))
            # print(f"Score {text}: {self._cosine_similarity(query_vec, vec)}")
        scored.sort(key=lambda x: x[1], reverse=True)
        return [text for text, _ in scored[:top_k]]

    def query(
        self,
        user_question: str,
        depth: int = 1,
        max_hops: int = 3,
        top_k_paths: int = 2,
        anchor_per_entity: int = 3,
        max_anchors: int = 10,
        neighbor_top_k: int = 4,
        neighbor_candidate_multiplier: int = 3,
        path_candidate_multiplier: int = 3,
        d: Optional[int] = None
    ):
        print(f"\nQuestion: {user_question}")

        if d is not None:
            depth = d
        
        raw_extraction = self.entity_chain.invoke({"question": user_question})
        target_entities = self._clean_entities(raw_extraction)
        print(f"Entities: {target_entities}")
        
        if not target_entities:
            return "Không trích xuất được thực thể nào."

        found_anchors = self._search_anchor_nodes(
            target_entities,
            per_entity_k=anchor_per_entity,
            max_anchors=max_anchors
        )
        print(f"Anchor nodes: {found_anchors}")

        if not found_anchors:
            return "Không tìm thấy node nào trong Graph."

        neighbor_candidate_limit = max(neighbor_top_k * neighbor_candidate_multiplier, neighbor_top_k)
        neighbor_triplets = self._collect_neighbor_triplets(found_anchors, depth, max_edges=neighbor_candidate_limit)
        neighbor_triplets = self._rerank_texts(list(neighbor_triplets), user_question, top_k=neighbor_top_k)

        path_candidate_limit = max(top_k_paths * path_candidate_multiplier, top_k_paths)
        multi_hop_paths = self._find_multi_hop_paths(found_anchors, max_hops=max_hops, candidate_limit=path_candidate_limit)
        multi_hop_paths = self._rerank_texts(multi_hop_paths, user_question, top_k=top_k_paths)

        if not neighbor_triplets and not multi_hop_paths:
            return "Tìm thấy node nhưng không có thông tin liên kết."

        context_sections = []
        if neighbor_triplets:
            formatted_neighbors = "\n".join(f"- {edge}" for edge in neighbor_triplets)
            context_sections.append("Liên kết lân cận:\n" + formatted_neighbors)
        if multi_hop_paths:
            formatted_paths = "\n".join(f"- Đường: {path}" for path in multi_hop_paths)
            context_sections.append(f"Đường đi multi-hop (<= {max_hops} bước):\n" + formatted_paths)

        context_text = f"\n\n".join(context_sections)
        context_text = context_text.replace("--", " quan hệ ").replace("-->", " là ")
        
        print(f"Context ném vào LLM:\n{context_text}")

        answer_prompt = f"""<|im_start|>system
        Bạn là một máy trả lời câu hỏi chính xác. Chỉ sử dụng thông tin trong phần 'Dữ liệu' để trả lời.
        <|im_end|>
        <|im_start|>user
        Dữ liệu:
        {context_text}

        Câu hỏi: {user_question}
        Câu trả lời ngắn gọn:
        <|im_end|>
        <|im_start|>assistant
"""
        # Lưu ý: invoke của HuggingFacePipeline nhận string là ok
        return self.llm.invoke(answer_prompt)

if __name__ == "__main__":
    tiny_llm = load_tiny_vietnamese_llm()
    rag_system = SmartGraphRAG(llm_model=tiny_llm)

    rag_system.add_triplet("Sơn Tùng MTP", "Thái Bình", "sinh ra tại")
    rag_system.add_triplet("Sơn Tùng MTP", "Sky Tour", "tổ chức")
    rag_system.build_vector("Sơn Tùng MTP", "Thái Bình", "sinh ra tại")

    result = rag_system.query("Sơn Tùng MTP quê ở đâu?")
    print(f"Answer: {result}")
