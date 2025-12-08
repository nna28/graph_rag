import argparse
from GraphBuilder.builder import Graph
from dotenv import load_dotenv

if __name__ == "__main__":
    
    load_dotenv()
    
    g: Graph = Graph.load()
    
    g.build()