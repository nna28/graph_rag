
from typing import Set, Dict, Tuple



class Graph:
    
    edges: Set[Tuple[str, str, str]] = set()
    nodes: Dict[str, Dict[str, str]] = {}
    
    
    def build(self):
        from GraphBuilder.db.models import PERSON_RELATIONSHIP_LABEL_TO_ATTRIBUTE,  ENTITY_TYPE_TO_CLASS, connect_by_name
        for edge in self.edges:
            try:
                relationship_type = PERSON_RELATIONSHIP_LABEL_TO_ATTRIBUTE[edge[2]]
                src_entity = self.nodes[edge[0]]
                des_entity = self.nodes[edge[1]]
                print(f"Src: {src_entity["type"]}, {src_entity["name"]}")
                print(f"Des: {des_entity["type"]}, {des_entity["name"]}")
                
                src_class = ENTITY_TYPE_TO_CLASS[src_entity["type"]]
                des_class = ENTITY_TYPE_TO_CLASS[des_entity["type"]]
                
                if not src_class.nodes.get_or_none(name = src_entity["name"]):
                    src_obj = src_class(
                        name = src_entity["name"],
                        link = edge[0]
                    ).save()
                    
                if not des_class.nodes.get_or_none(name=des_entity["name"]):
                    des_obj = ENTITY_TYPE_TO_CLASS[des_entity["type"]](
                        name = des_entity["name"],
                        link = edge[1]
                    ).save()
                connect_by_name(edge[0], edge[1], relationship_type)
            except Exception as e:
                print(e)
                # pass
            
            
    
    @classmethod
    def load(cls):
        import pandas as pd
        
        obj = cls()
        
        df_nodes = pd.read_csv("nodes.csv")
        df_edges = pd.read_csv("edges.csv")
        
        obj.edges = {
            (item["src"], item["des"], item["type"]) for _, item in df_edges.iterrows()
        }
        
        obj.nodes = {
            item["link"]: {
                "name" : item["name"],
                "type" : item["type"]    
            }for _, item in df_nodes.iterrows()
        }
        return obj