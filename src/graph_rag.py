import networkx as nx
import torch
import re
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
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=100, 
        temperature=0.01,   
        repetition_penalty=1.2,
        do_sample=False,    
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
            template="""Trích xuất Danh từ riêng (Thực thể) chính từ câu hỏi.
                Câu: Sơn Tùng MTP quê ở đâu?
                Thực thể: Sơn Tùng MTP

                Câu: Tập đoàn FPT thành lập năm nào?
                Thực thể: FPT

                Câu: Ai là người sáng lập Microsoft?
                Thực thể: Microsoft

                Câu: {question}
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

    def query(self, user_question: str, depth: int = 1):
        print(f"\nQuestion: {user_question}")
        
        raw_extraction = self.entity_chain.invoke({"question": user_question})
        target_entities = self._clean_entities(raw_extraction)
        print(f"Entities: {target_entities}")
        
        if not target_entities:
            return "Không trích xuất được thực thể nào."

        found_anchors = set()
        for entity in target_entities:
            results = self.vector_store.similarity_search_with_score(entity, k=3)
            for doc, score in results:
                found_anchors.add(doc.page_content)
        
        if not found_anchors:
            return "Không tìm thấy node nào trong Graph."

        context_triplets = set()
        for node in found_anchors:
            if not self.graph.has_node(node): continue
            
            subgraph = nx.ego_graph(self.graph, node, radius=depth)
            for u, v, data in subgraph.edges(data=True):
                rel = data.get('relation', 'liên quan')
                context_triplets.add(f"- {u} {rel} {v}")

        context_text = "\n".join(context_triplets)
        
        if not context_text:
            return "Tìm thấy node nhưng không có thông tin liên kết."

        answer_prompt = f"""Thông tin:
{context_text}

Hỏi: {user_question}
Trả lời ngắn gọn:"""
        
        return self.llm.invoke(answer_prompt)

if __name__ == "__main__":
    tiny_llm = load_tiny_vietnamese_llm()
    rag_system = SmartGraphRAG(llm_model=tiny_llm)

    rag_system.add_triplet("Sơn Tùng MTP", "Thái Bình", "sinh ra tại")
    rag_system.add_triplet("Sơn Tùng MTP", "Sky Tour", "tổ chức")
    rag_system.build_vector("Sơn Tùng MTP", "Thái Bình", "sinh ra tại")

    result = rag_system.query("Sơn Tùng MTP quê ở đâu?")
    print(f"Answer: {result}")