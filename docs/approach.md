# Báo cáo quá trình thực hiện


## Tech stack

- BeautifulSoup4: Trích xuất dữ liệu từ các html được lấy về.
- Neo4j (neomodel): Lưu trữ đồ thị được tạo.

## Thuật toán duyệt

Nhận các hạt giống đầu tiên là các nhà khoa học có giải nobel.

[Danh sách nhà khoa học đạt giải nobel](/wiki/Danh_s%C3%A1ch_ng%C6%B0%E1%BB%9Di_%C4%91o%E1%BA%A1t_gi%E1%BA%A3i_Nobel)

Thực hiện duyệt BFS.

Vì lý do ưu hiện độ rộng của đồ thị và các liên kết giữa các nhà khoa học. Cho nên việc thực hiện duyệt BFS giúp đảm bảo các tiêu chi trên


```python


def processed(link: str) -> List[str]:
    """
    Lấy dữ liệu từ infobox của các nhà khoa học
    """

while (not self.q.isEmpty()):
    link = self.q.dequeue()
    self.visited.add(link)
    links = self.processed(link)

    for new_link in links:
        if not self.visited.exist(new_link):
            self.q.enqueue(new_link)
```

Để duyệt ta tạo cạnh là liên kết giữa 2 link gốc và thông tin trong infobox của nhà khoa học. 

![infobox](../statics/infobox.png)

## Cấu trúc dữ liệu

Thực hiện lưu cạnh và nodes dựa trên cấu trúc:

```python
edges: Set[Tuple[str, str, str]] = set()
nodes: Dict[str, Dict[str, str]] = {}
```

Cạnh là tuple của (link_sorces, link_destination, edge_type).
Nodes là Dict lưu với link là key và meta-data là value để giống từ cạnh đến node lấy dữ liệu quan trọng của node.

## Vấn đề gặp phải

### Dữ liệu quá nhiễu khi không giới nhận loại cạnh:

Khi không giới hạn các loại cạnh, việc duyệt luôn cả các thực thể không liên quan đến các nhà khoa học sẽ tạo ra nhiều cạnh và thực thể không liên quan dẫn đến dữ liệu trả về bị **nhiễu**.

**Giải pháp**: Giới hạn một vài cạnh được xuất hiện nhiều hoặc chỉ xuất hiện trong các infobox của nhà khoa học. 

```python
EDGE_TYPES = [
    "Giải thưởng",
    "Sinh",
    "Nơi công tác",
    "Học vị",
    "Các sinh viên nổi tiếng",
    "Tài liệu",
    "Luận án",
    "Giáo dục",
    "Người hướng dẫn luận án tiến sĩ",
    "Mất",
    "Trường lớp",
    "Các nghiên cứu sinh nổi tiếng",
    "Nổi tiếng vì",
    "Phối ngẫu",
    "Giải thưởng nổi bật",
    "Nghề nghiệp",
    "Cố vấn nghiên cứu khác",
    "Ảnh hưởng bởi",
    "Ảnh hưởng tới",
    "Hôn nhân", 
    "Con cái",
    "Quốc tịch"
    
]
```

Giảm nhiễu và tạo được mạng có ý nghĩa.

