import networkx as nx
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

class SmartGraphRAG:
    def __init__(self, llm_model):
        self.llm = llm_model
        self.graph = nx.DiGraph()
        
        self.embedding_model = HuggingFaceEmbeddings(model_name="bkai-foundation-models/vietnamese-bi-encoder")
        print("Done!!")
        self.vector_store = Chroma(
            collection_name="graph_nodes",
            embedding_function=self.embedding_model,
            persist_directory="chroma"
        )
        self.visited_nodes = set() 

        self.entity_extract_prompt = PromptTemplate(
            template="""Bạn là một chuyên gia trích xuất thông tin. Nhiệm vụ của bạn là xác định các Thực thể chính (Người, Tổ chức, Địa điểm, Khái niệm) trong câu hỏi để dùng làm từ khóa tìm kiếm.
            
            Yêu cầu:
            - Chỉ trích xuất danh từ riêng hoặc cụm danh từ quan trọng.
            - Bỏ qua các từ để hỏi (ai, là gì, ở đâu...).
            - Trả về kết quả ngăn cách bởi dấu phẩy. Không giải thích thêm.
            
            Ví dụ: 
            Hỏi: Ông A làm việc ở đâu? -> Ông A
            Hỏi: VNPT có trụ sở tại tỉnh nào? -> VNPT
            
            Câu hỏi: {question}
            Kết quả:""",
            input_variables=["question"]
        )
        self.entity_chain = self.entity_extract_prompt | self.llm | StrOutputParser()

    def add_triplet(self, subj: str, obj: str, rel: str):
        self.graph.add_edge(subj, obj, relation=rel)
        # self.build_vector(subj, obj, rel)

    def build_vector(self, subj: str, obj: str, rel: str):
        if subj not in self.visited_nodes:
            self.vector_store.add_documents([Document(page_content=subj, metadata={"type": "node"})])
            self.visited_nodes.add(subj)
            
        if obj not in self.visited_nodes:
            self.vector_store.add_documents([Document(page_content=obj, metadata={"type": "node"})])
            self.visited_nodes.add(obj)

    def query(self, user_question: str, depth: int = 1):
        print(f"\nQuestion: {user_question}")
        
        extracted_entities_str = self.entity_chain.invoke({"question": user_question})
        target_entities = [e.strip() for e in extracted_entities_str.split(',') if e.strip()]
        
        print(f"Extracted Entities: {target_entities}")
        
        if not target_entities:
            return "Không trích xuất được thực thể nào từ câu hỏi."

        found_anchors = set()
        for entity in target_entities:
            results = self.vector_store.similarity_search_with_score(entity, k=2)
            
            for doc, score in results:
                print(f"   Vector Match: '{entity}' -> '{doc.page_content}' (Score: {score:.4f})")
                found_anchors.add(doc.page_content)
        
        if not found_anchors:
            return "Không tìm thấy node nào trong Graph khớp với thực thể."

        print(f"Final Anchor Nodes: {found_anchors} | Depth: {depth}")
        
        context_triplets = set()
        for node in found_anchors:
            if not self.graph.has_node(node): continue
            
            subgraph = nx.ego_graph(self.graph, node, radius=depth)
            
            for u, v, data in subgraph.edges(data=True):
                rel = data.get('relation', 'liên quan tới')
                context_triplets.add(f"{u} {rel} {v}")

        context_text = "\n".join(context_triplets)
        print(f"Graph Context ({len(context_triplets)} triplets):\n{context_text}")
        
        if not context_text:
            return "Tìm thấy node nhưng không có thông tin liên kết."

        answer_prompt = f"""Dựa vào tri thức sau từ Knowledge Graph, hãy trả lời câu hỏi.
        
        Tri thức:
        {context_text}
        
        Câu hỏi: {user_question}
        Trả lời:"""
        
        return self.llm.invoke(answer_prompt).content