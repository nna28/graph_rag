from src.graph_rag import SmartGraphRAG
from langchain_google_genai import ChatGoogleGenerativeAI
import pandas as pd


if __name__ == "__main__":

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0,
        convert_system_message_to_human=True
    )

    rag = SmartGraphRAG(llm_model=llm)
    # df = pd.read_csv("/Users/huynhnguyen/WorkDir/Network/data/final/edges.csv")
    # for i, row in df.iterrows():
    #     rag.build_vector(row["src"], row["des"], row["type"])
    # print(rag.query("Aage Bohr có những giải thưởng gì?"))
    for item in rag.vector_store.similarity_search_with_score("Obama", k=10):
        print(item)

