import re
import json
import yaml
import pypred
import networkx as nx
import xml.etree.ElementTree as ET
from openai import AzureOpenAI, OpenAI
from typing import Any, Dict, Text, TypedDict
from termcolor import colored
from jinja2 import Environment


class ReprMixin:
    def __repr__(self):
        return f"<{self.__class__.__name__} at {hex(id(self))}>"


def char_to_ascii(value: Text):
    return ord(value)


def ascii_to_char(value: int):
    return chr(value)


def extract_numbers(text):
    pattern = r"\d+"
    matches = re.findall(pattern, text)
    numbers = [int(match) for match in matches]
    return numbers


environment = Environment()
environment.add_extension("jinja2.ext.debug")
environment.filters["char_to_ascii"] = char_to_ascii
environment.filters["ascii_to_char"] = ascii_to_char
environment.filters["extract_numbers"] = extract_numbers


def render(template, **kwargs):
    t = environment.from_string(template)
    return t.render(**kwargs)


def predicate(condition, **kwargs):
    condition = render(condition, **kwargs)
    p = pypred.Predicate(condition)
    res, ctx = p.analyze(kwargs)
    return res


def load_from_graphml(path: Text) -> Dict:
    graph = nx.read_graphml(path)

    tree = ET.parse(path)
    root = tree.getroot()

    # 命名空间
    ns = {"graphml": "http://graphml.graphdrawing.org/xmlns"}

    # 查找所有的 <key> 元素并存储图级别的 key
    graph_keys = {}
    default_values = {}
    for key in root.findall("graphml:key", ns):
        if key.attrib.get("for") == "graph":
            key_id = key.attrib["id"]
            key_name = key.attrib.get("attr.name")
            graph_keys[key_id] = key_name
            default = key.find("graphml:default", ns)
            if default is not None:
                default_values[key_id] = default.text

    # 查找 <graph> 元素
    graph_element = root.find("graphml:graph", ns)

    # 提取图级别的属性
    graph_attributes = {}
    for data in graph_element.findall("graphml:data", ns):
        key = data.attrib["key"]
        if key in graph_keys:
            attribute_name = graph_keys[key]
            attribute_value = data.text
            graph_attributes[attribute_name] = attribute_value

    for key_id, default_value in default_values.items():
        attribute_name = graph_keys[key_id]
        if attribute_name not in graph_attributes:
            graph_attributes[attribute_name] = default_value

    graph_dict = nx.node_link_data(graph)
    graph_dict["graph"]["graph_default"] = graph_attributes

    nodes = {}
    conditional_nodes = set()
    for node in graph_dict["nodes"]:
        nodes[node["id"]] = node["label"]
        if "next" in node:
            conditional_nodes.add(node["id"])
            node["next"] = json.loads(node["next"].strip())

    edges = []
    for edge in graph_dict["links"]:
        source_node = edge["source"]
        if source_node in conditional_nodes:
            # print(f"ignore: {edge}")
            continue
        target_node = edge["target"]
        edges.append({"source": nodes[edge["source"]], "target": nodes[edge["target"]]})

    graph_dict_final = {}
    graph_dict_final["links"] = edges
    graph_dict_final["nodes"] = graph_dict["nodes"]
    graph_dict_final["graph"] = graph_dict["graph"]

    yaml_path_ori = path.replace(".graphml", "_ori.yaml")
    with open(yaml_path_ori, "w") as f:
        yaml.dump(graph_dict, f, default_flow_style=False, allow_unicode=True)

    yaml_path = path.replace(".graphml", ".yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(graph_dict_final, f, default_flow_style=False, allow_unicode=True)

    return graph_dict_final


def chat(prompt, model=None):
    if model == "claude":
        client = OpenAI(
            api_key="xxx",
            base_url="xx",
        )
        completion = client.chat.completions.create(
            model="claude-3-5-sonnet",
            messages=[{"role": "user", "content": prompt}],
        )
        res = completion.choices[0].message.content
    else:
        client = AzureOpenAI(
            api_key="xxx",
            api_version="2024-02-01",
            azure_endpoint="xx",
        )
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        res = completion.choices[0].message.content
    return res


def function_exists(module, func_name):
    return hasattr(module, func_name) and callable(getattr(module, func_name))


def create_dynamic_class(class_name: Text, attrs: Dict[str, Any]):
    TypedDictClass = TypedDict(class_name + "Dict", attrs)

    class DynamicClass(TypedDictClass):
        def to_typed_dict(self):
            return {name: getattr(self, name) for name in attrs.keys()}

    DynamicClass = type(class_name, (DynamicClass,), {})
    DynamicClass.__annotations__ = attrs

    return DynamicClass
