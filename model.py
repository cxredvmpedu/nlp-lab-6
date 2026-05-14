from pydantic.dataclasses import dataclass


@dataclass
class PropertyItem:
    platform: str
    category: str
    url: str
    desc: str
