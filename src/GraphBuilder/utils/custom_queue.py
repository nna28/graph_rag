

class CustomQueue:
  def __init__(self):
    self.q = []
    
  def enqueue(self, element):
    self.q.append(element)

  def dequeue(self):
    if self.isEmpty():
      return "Queue is empty"
    return self.q.pop(0)

  def peek(self):
    if self.isEmpty():
      return "Queue is empty"
    return self.q[0]

  def isEmpty(self):
    return len(self.q) == 0

  def size(self):
    return len(self.q)

  def serialize(self):
    import json
    with open("queue.json", "w", encoding="utf-8") as f:
      json.dump(self.q, f, indent=2, ensure_ascii=False)
    
  @classmethod
  def unserialize(cls, path):
    import json 
    
    obj = cls()
    
    with open("queue.json", "r") as f:
      obj.q = json.load(f)
      
    return obj
      
      
      
  # if __name__ == "__main__":
    # from GraphBuilder.utils.custom_queue import CustomQueue
    # q: CustomQueue = CustomQueue()
    
    # q.enqueue("test")
    # q.serialize("test.json")
    
    # q: CustomQueue = CustomQueue.unserialize("test.json")
    
    # print(q.peek())
    
    