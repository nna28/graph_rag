from typing import Dict, Set, Type

from neomodel import (
    config,
    StructuredNode,
    StringProperty,
    UniqueIdProperty,
    RelationshipTo,
    db
)
import os
from dotenv import load_dotenv

load_dotenv()


config.DATABASE_URL = f'bolt://{os.environ["NEO4J_USER"]}:{os.environ["NEO4J_PASSWORD"]}@localhost:7687'  # default


class Entity(StructuredNode):
    """
    Base node with the shared wiki link and display name metadata used for
    every entity that appears in the knowledge graph.
    """

    uid = UniqueIdProperty()
    name = StringProperty(unique_index=True, required=True)
    link = StringProperty(unique_index=True, required=False)


class UnknownEntity(Entity):
    """
    Placeholder node type when we cannot confidently classify the target.
    """

    pass


class Country(Entity):
    code = StringProperty(unique_index=True, required=False)


class Field(Entity):
    pass


class Organization(Entity):
    pass


class University(Organization):
    pass


class Article(Entity):
    pass


class Award(Entity):
    pass


class Job(Entity):
    pass


class Place(Entity):
    pass


class Event(Entity):
    pass


class Person(Entity):
    """
    Person node that is the primary subject in the graph. Specific outgoing
    relationships are attached dynamically below using the schema map.
    """

    pass


PERSON_RELATIONSHIP_SCHEMAS: Dict[str, Dict[str, object]] = {
    "died": {
        "label": "Mất",
        "relationship_type": "MAT",
        "targets": {"UNKNOWN", "PLACE", "COUNTRY", "PERSON", "ORGANIZATION"},
    },
    "schools": {
        "label": "Trường lớp",
        "relationship_type": "TRUONG_LOP",
        "targets": {"UNIVERSITY", "PLACE", "ORGANIZATION", "UNKNOWN"},
    },
    "nationalities": {
        "label": "Quốc tịch",
        "relationship_type": "QUOC_TICH",
        "targets": {"COUNTRY", "PLACE", "UNKNOWN", "PERSON", "ORGANIZATION"},
    },
    "notable_for": {
        "label": "Nổi tiếng vì",
        "relationship_type": "NOI_TIENG_VI",
        "targets": {"ARTICLE", "PLACE", "ORGANIZATION", "UNKNOWN"},
    },
    "influenced": {
        "label": "Ảnh hưởng tới",
        "relationship_type": "ANH_HUONG_TOI",
        "targets": {"FIELD", "PERSON", "UNKNOWN"},
    },
    "education": {
        "label": "Giáo dục",
        "relationship_type": "GIAO_DUC",
        "targets": {"ORGANIZATION", "UNKNOWN"},
    },
    "birth_contexts": {
        "label": "Sinh",
        "relationship_type": "SINH",
        "targets": {"UNKNOWN", "COUNTRY", "PLACE", "ORGANIZATION", "EVENT", "PERSON"},
    },
    "awards": {
        "label": "Giải thưởng",
        "relationship_type": "GIAI_THUONG",
        "targets": {"AWARD", "PERSON", "PLACE", "ORGANIZATION", "EVENT", "UNKNOWN"},
    },
    "occupations": {
        "label": "Nghề nghiệp",
        "relationship_type": "NGHE_NGHIEP",
        "targets": {"JOB", "PERSON", "ORGANIZATION", "UNKNOWN"},
    },
    "workplaces": {
        "label": "Nơi công tác",
        "relationship_type": "NOI_CONG_TAC",
        "targets": {"PLACE", "ORGANIZATION", "UNKNOWN"},
    },
    "major_awards": {
        "label": "Giải thưởng nổi bật",
        "relationship_type": "GIAI_THUONG_NOI_BAT",
        "targets": {"AWARD", "PLACE", "ORGANIZATION", "EVENT", "UNKNOWN"},
    },
    "influenced_by": {
        "label": "Ảnh hưởng bởi",
        "relationship_type": "ANH_HUONG_BOI",
        "targets": {"UNKNOWN", "PERSON"},
    },
    "children": {
        "label": "Con cái",
        "relationship_type": "CON_CAI",
        "targets": {"PERSON", "UNKNOWN"},
    },
    "degrees": {
        "label": "Học vị",
        "relationship_type": "HOC_VI",
        "targets": {"ORGANIZATION", "UNKNOWN"},
    },
    "notable_doctoral_students": {
        "label": "Các nghiên cứu sinh nổi tiếng",
        "relationship_type": "CAC_NGHIEN_CUU_SINH_NOI_TIENG",
        "targets": {"UNKNOWN", "PERSON"},
    },
    "marriages": {
        "label": "Hôn nhân",
        "relationship_type": "HON_NHAN",
        "targets": {"PERSON"},
    },
    "doctoral_advisors": {
        "label": "Người hướng dẫn luận án tiến sĩ",
        "relationship_type": "NGUOI_HUONG_DAN_LUAN_AN_TIEN_SI",
        "targets": {"UNKNOWN", "PERSON"},
    },
    "spouses": {
        "label": "Phối ngẫu",
        "relationship_type": "PHOI_NGAU",
        "targets": {"UNKNOWN", "PERSON"},
    },
    "notable_students": {
        "label": "Các sinh viên nổi tiếng",
        "relationship_type": "CAC_SINH_VIEN_NOI_TIENG",
        "targets": {"UNKNOWN", "PERSON"},
    },
    "theses": {
        "label": "Luận án",
        "relationship_type": "LUAN_AN",
        "targets": {"UNKNOWN"},
    },
}

for attr_name, schema in PERSON_RELATIONSHIP_SCHEMAS.items():
    setattr(Person, attr_name, RelationshipTo("Entity", schema["relationship_type"]))


PERSON_RELATIONSHIP_LABEL_TO_ATTRIBUTE: Dict[str, str] = {
    schema["label"]: attr_name for attr_name, schema in PERSON_RELATIONSHIP_SCHEMAS.items()
}


ENTITY_TYPE_TO_CLASS: Dict[str, Type[Entity]] = {
    "PERSON": Person,
    "UNKNOWN": UnknownEntity,
    "UNIVERSITY": University,
    "COUNTRY": Country,
    "ARTICLE": Article,
    "FIELD": Field,
    "ORGANIZATION": Organization,
    "AWARD": Award,
    "JOB": Job,
    "PLACE": Place,
    "EVENT": Event,
}

def connect_by_name(src, dst, rel_type):
    db.cypher_query(f"""
        MATCH (s {{link: $src}}), (d {{link: $dst}})
        MERGE (s)-[:{rel_type}]->(d)
    """, {"src": src, "dst": dst})