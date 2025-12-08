
from typing import Dict, List, Tuple, Any, Set
import requests
from bs4 import BeautifulSoup
from GraphBuilder.utils.custom_queue import CustomQueue
from GraphBuilder.utils.custom_set import CustomSet
from collections import defaultdict
from GraphBuilder.utils.utils import get_infobox, get_name


my_headers = {
    'User-Agent': 'ScienceNetworkBot/1.0 (mailto:hnc203204@gmail.com)'
}

def characterize_type_by_edge(labels: Set[str]) -> str:
    """
    Heuristically classify an entity type using its infobox edge labels.
    Returns one of: PERSON, ORGANIZATION, PLACE, WORK, AWARD, EVENT, UNKNOWN
    """
    if not labels:
        return "UNKNOWN"

    # normalize to lowercase
    norm = {str(l).strip().lower() for l in labels if l}

    
    person_keys = {
        "sinh", "sinh ngày", "mất", "ngày sinh", "ngày mất", "năm sinh", "năm mất",
        "nghề nghiệp", "quốc tịch", "vợ/chồng", "vợ", "chồng", "con", "cha mẹ",
        "học vị", "học vấn", "alma mater", "nơi sinh", "nơi mất", "giải thưởng",
        "born", "birth date", "birth place", "died", "death date", "death place",
        "occupation", "nationality", "spouse", "children", "parents", "education"
    }
    organization_keys = {
        "thành lập", "người sáng lập", "trụ sở", "lĩnh vực", "cơ quan chủ quản",
        "nhân viên", "thành viên", "hiệu trưởng", "sinh viên", "doanh thu",
        "founded", "founder", "headquarters", "industry", "parent organization",
        "employees", "members", "revenue"
    }
    place_keys = {
        "quốc gia", "tỉnh", "thành phố", "huyện", "xã", "diện tích", "dân số",
        "mật độ", "vĩ độ", "kinh độ", "múi giờ", "độ cao",
        "country", "province", "city", "area", "population", "density",
        "coordinates", "latitude", "longitude", "timezone", "elevation"
    }
    work_keys = {
        "tác giả", "xuất bản", "phát hành", "nhà xuất bản", "thể loại",
        "đạo diễn", "diễn viên", "thời lượng", "ngôn ngữ gốc",
        "author", "publisher", "published", "release date", "genre",
        "director", "starring", "runtime", "original language"
    }
    award_keys = {
        "trao tặng bởi", "năm thành lập", "hạng mục", "người nhận đầu tiên",
        "điều kiện", "số tiền", "được trao cho",
        "awarded for", "presenter", "first awarded", "cash award", "category"
    }
    event_keys = {
        "thời gian", "địa điểm", "nguyên nhân", "kết quả", "tham gia",
        "bắt đầu", "kết thúc",
        "date", "location", "cause", "result", "participants", "start date", "end date"
    }

    buckets = {
        "PERSON": person_keys,
        "ORGANIZATION": organization_keys,
        "PLACE": place_keys,
        "WORK": work_keys,
        "AWARD": award_keys,
        "EVENT": event_keys,
    }

    scores: Dict[str, int] = {k: 0 for k in buckets}
    for lab in norm:
        for typ, keys in buckets.items():
            
            if lab in keys:
                scores[typ] += 2
            
            for k in keys:
                if k in lab and k != lab:
                    scores[typ] += 1

    # Strong signals to break ties quickly
    strong_person = any(x in norm for x in {"sinh", "ngày sinh", "birth date", "occupation"})
    strong_org = any(x in norm for x in {"thành lập", "founded", "headquarters"})
    strong_place = any(x in norm for x in {"dân số", "diện tích", "population", "area", "coordinates"})

    best_type = max(scores.items(), key=lambda kv: kv[1])[0]
    best_score = scores[best_type]

    if best_score == 0:
        return "PERSON" if strong_person else ("ORGANIZATION" if strong_org else ("PLACE" if strong_place else "UNKNOWN"))
    return best_type
        
def isPerson(link) -> bool:
    """
    Determine whether the Wikipedia page behind `link` describes a person.
    The decision is primarily driven by infobox signals with a fallback to
    category heuristics when an infobox is missing or ambiguous.
    """
    
    print(link)
    try:
        page = requests.get(
            f"https://vi.wikipedia.org{link}",
            headers=my_headers,
            timeout=10,
        )
        page.raise_for_status()
    except requests.RequestException as exc:
        print(f"Failed to fetch {link}: {exc}")
        return False

    soup = BeautifulSoup(page.content, "html.parser")

    name = get_name(soup)
    if not name:
        return False

    node_labels: Set[str] = set()
    ent_type = "UNKNOWN"

    if (info := get_infobox(soup)):
        for th in info.select("tr > th.infobox-label"):
            label_text = th.get_text(strip=True)
            if label_text:
                node_labels.add(label_text)

        ent_type = characterize_type_by_edge(node_labels)
        if ent_type == "PERSON":
            return True
        if ent_type in {"ORGANIZATION", "PLACE"}:
            return False

    # Fallback: use category section heuristics when infobox data is missing/inconclusive.
    category_block = soup.find("div", id="mw-normal-catlinks")
    if category_block:
        categories = [
            a.get_text(strip=True).lower()
            for a in category_block.select("li > a")
        ]
        person_markers = {"nhà ", "người", "sinh", "births", "nhân vật"}
        non_person_markers = {"công ty", "thành phố", "quốc gia", "tỉnh", "tác phẩm"}

        if any(any(marker in cat for marker in person_markers) for cat in categories):
            return True
        if any(any(marker in cat for marker in non_person_markers) for cat in categories):
            return False

    # With no strong signals we conservatively assume non-person.
    return False
    
labels = [
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


class ExtractWiki:
    
    edges: Set[Tuple[str, str, str]] = set()
    nodes: Dict[str, Dict[str, str]] = {}
    q: CustomQueue = CustomQueue()
    visited: CustomSet = CustomSet()
        
    def traversal(self, seeds):
        

        for link in seeds:
            # print(link)
            self.q.enqueue(link)
        
        # visited: Dict[str, bool] = defaultdict(lambda: False)

        
        while (not self.q.isEmpty()) or len(self.nodes.keys()) <= 2000:
            link = self.q.dequeue()
            # print(link)
            self.visited.add(link)
            
            try:
                links = self.processed(link)
            except Exception as e:
                print(e)
                links = []
            
            for new_link in links:
                if not self.visited.exist(new_link):
                    self.q.enqueue(new_link)
    
    def processed(self, link) -> List[str]:
        page = requests.get(f"https://vi.wikipedia.org{link}", headers=my_headers)

        soup = BeautifulSoup(page.content, "html.parser")

        name = get_name(soup)
        if not name:
            return []

        
        all_link: List[str] = []

        
        node_labels: Set[str] = set()
        print(link, name)
        if (info := get_infobox(soup)):
            # print("1")
            for tr in info.find_all("tr"):
                if (edge := tr.find("th", class_ = "infobox-label")):
                    label_text = edge.get_text(strip=True) if edge else None
                    if label_text:
                        node_labels.add(label_text)
                    print(label_text)
                    if not (label_text in labels):
                        continue
                    for a_ele in tr.find_all("a"):
                        if a_ele.string:
                            if (href := a_ele.get("href")):
                                print(f"\t Edges({link}, {href}, {label_text})")
                                self.edges.add((link, href, label_text))
                                # self.edges_types.add(label_text)
                                all_link.append(href)

        ent_type = characterize_type_by_edge(node_labels)
        self.nodes[link] = {
            "name": name,
            "type": ent_type
        }
        return all_link
    
    def serialize(self):
        import pandas as pd
        new_edges = pd.DataFrame({
            "src" : [t[0] for t in self.edges],
            "des" : [t[1] for t in self.edges],
            "type": [t[2] for t in self.edges]
        })
        
        try:
            edges_existing = pd.read_csv("edges.csv")
        except Exception as e:
            print(e)
            edges_existing = pd.DataFrame({
                "src" : [],
                "des" : [],
                "type": []
            })
        combined = pd.concat([edges_existing, new_edges], ignore_index=True)
        combined.to_csv("edges.csv", index=False)
        
        new_nodes = pd.DataFrame({
            "link" : [item for item in self.nodes.keys()],
            "name" : [item["name"] for item in self.nodes.values()],
            "type" : [item["type"] for item in self.nodes.values()]
        })
        
        try:
            nodes_existing = pd.read_csv("nodes.csv")
        except Exception as e:
            print(e)
            nodes_existing = pd.DataFrame({
                "link" : [],
                "name" : [],
                "type" : []
            })
        node_combined = pd.concat([nodes_existing, new_nodes], ignore_index=True)
        node_combined.to_csv("nodes.csv", index=False)
        
        
    def save(self):
        self.q.serialize()
        self.visited.serialize()
        
    @classmethod
    def load(cls):
        obj = cls()
        
        obj.q = CustomQueue.unserialize()
        obj.visited = CustomSet.unserialize()
        
        return obj
    

    
    
            
        
    # hrefs = []
    # for table in wiki_tables:
    #     for link in table.find_all("a"):
    #         hrefs.append(link.get("href"))
    # return hrefs


def get_all_href(link) -> List[str]:
    page = requests.get(f"https://vi.wikipedia.org{link}", headers=my_headers)

    soup = BeautifulSoup(page.content, "html.parser")
    wiki_tables = soup.find_all("table", class_ = "wikitable")
    hrefs = []
    for table in wiki_tables:
        for link in table.find_all("a"):
            hrefs.append(link.get("href"))
    return hrefs



if __name__ == "__main__":
    
    seeds = [
        "/wiki/Danh_s%C3%A1ch_ng%C6%B0%E1%BB%9Di_%C4%91o%E1%BA%A1t_gi%E1%BA%A3i_Nobel"
    ]
    
    extracter = ExtractWiki()
    
    seeds = [link for seed in seeds for link in get_all_href(seed)]
    # for link in seeds:
    #     print(link)
    try:
        extracter.traversal(seeds)
    except Exception as e:
        print(e)
        extracter.save()
        # extracter.serialize()
    finally:
        extracter.serialize()
    # page = requests.get("https://vi.wikipedia.org/wiki/Guglielmo_Marconi", headers=my_headers)
    # soup = BeautifulSoup(page.content, "html.parser")
    # print(get_name(soup))




