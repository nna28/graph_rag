
from typing import List, Dict
from bs4 import BeautifulSoup

def read_csv(path: str) -> List[Dict[str, str]]:
    import pandas as pd
    df = pd.read_csv(path)
    data = [
        {
            "name": row[1]["name"],
            "field": row[1]["field"],
            "country": row[1]["country"]
        }
             for row in df.iterrows()
    ]
    return data

def is_date(date: str) -> bool:
    import datetime
    date = date.strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y", "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%m/%d/%Y", "%Y"):
        try:
        
            datetime.strptime(date, fmt)
            return True
        except ValueError:
            continue
    return False


def get_infobox(soup: BeautifulSoup) -> BeautifulSoup:
    # print(data.prettify())
    for e in soup.find_all("table"):
        if e.get("class"):
            if "infobox" in e.get("class"):
                return e


    return None

def get_name(soup: BeautifulSoup) -> str:
    try:
        
        name = soup.find("h1").string
        # print(name)
        return name
    except Exception as e:
        print(e)
        return None
    # return soup.find("h1").string if soup.find("h1").string else None


if __name__ == "__main__":
    print(is_date("(1937-07-20)"))

    