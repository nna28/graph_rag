from typing import Set

class CustomSet:
    
    s: Set[str] = set()
    
    def add(self, element: str):
        self.s.add(element)
        
    def exist(self, element) -> bool:
        return element in self.s
    
    def serialize(self):
        import json 
        
        with open("set.json", "w", encoding="utf-8") as f:
            json.dump(list(self.s), f, indent=2, ensure_ascii=False)
            
    @classmethod
    def unserialize(cls):
        import json 
        obj = cls()
        with open("set.json", "r") as f:
            obj.s = set(json.load(f))
        return obj
            

if __name__ == "__main__":
    my_set: CustomSet = CustomSet.unserialize()
    
    print(my_set.s) 
    
        
    