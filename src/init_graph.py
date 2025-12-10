
import pandas as pd
from src.graph_rag import SmartGraphRAG



def init(rag: SmartGraphRAG):
    df = pd.read_csv("/Users/huynhnguyen/WorkDir/Network/data/final/edges.csv")
    for i, row in df.iterrows():
        rag.add_triplet(row["src"], row["des"], row["type"])
        # rag.add_triplet(row["des"], row["src"], row["type"])
        
    print("Done!!")
