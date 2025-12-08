from src.graph_rag import SmartGraphRAG
from langchain_google_genai import ChatGoogleGenerativeAI
from src.init_graph import init



if __name__ == "__main__":

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0,
        convert_system_message_to_human=True
    )

    rag = SmartGraphRAG(llm_model=llm)
    print("Init!!")
    init(rag)
    
    print(rag.query("Aage Bohr có những giải thưởng gì?"))
    

