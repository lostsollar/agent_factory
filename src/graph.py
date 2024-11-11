import json
import logging
from functools import partial
import re
from types import ModuleType
from typing import (
    Annotated,
    Dict,
    List,
    Optional,
    Set,
    Text,
    Union,
)
from langgraph.checkpoint import MemorySaver, BaseCheckpointSaver
from langchain_core.messages import AnyMessage
from langgraph.graph import StateGraph, add_messages
from termcolor import colored
from operator import add

from . import nodes as default_nodes
from .utils import chat, create_dynamic_class, function_exists, predicate, render


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


TYPE_DICT = {
    "Integer": int,
    "Bool": bool,
    "Text": Text,
    "List": List,
    "Dict": Dict,
}


class Graph:
    def __init__(
        self,
        config: Dict,
        node_module: ModuleType = None,
        edge_module: ModuleType = None,
        type_dict: Dict = {},
        chat_func: callable = chat,
        saver: BaseCheckpointSaver = MemorySaver(),
    ):
        self.node_module = node_module
        self.edge_module = edge_module
        self.type_dict = TYPE_DICT | type_dict
        self.chat = chat_func

        if "main" in config:
            scheduler = self.build_graph_scheduler(config)
            logger.info(colored(f"Scheduler: {scheduler}", "red"))

            graphs = {}
            for graph in scheduler:
                logger.info(colored(f"Graph [{graph}] build start!"))
                graphs[graph] = self.build_graph(config[graph], graphs, saver=saver)
                logger.info(colored(f"Graph [{graph}] build completed!"))

            self.graph = graphs["main"]
        else:
            self.graph = self.build_graph(config, saver=saver)

        logger.info(colored("Graph build success!", "green"))

    def build_graph_scheduler(self, config: Dict) -> List[Text]:
        graph_edges = {}
        for graph_n, graph_c in config.items():
            links = graph_c.get("links", {})
            targets = set([link["target"] for link in links])

            for node in graph_c["nodes"]:
                if "next" not in node:
                    continue

                targets.update(self._next_path_map(node["next"]).keys())

            graph_edges[graph_n] = targets

        scheduler = []
        graphs_cnt = len(graph_edges.keys())
        for _ in range(graphs_cnt):
            for graph_n, targets in graph_edges.items():
                if graph_n in scheduler:
                    continue
                if not targets or all(
                    [x in scheduler or x not in graph_edges.keys() for x in targets]
                ):
                    scheduler.append(graph_n)
            if len(scheduler) == graphs_cnt:
                break

        assert len(scheduler) == graphs_cnt

        return scheduler

    def build_graph(
        self,
        config: Dict,
        compiled_nodes: Dict = {},
        saver: BaseCheckpointSaver = MemorySaver(),
    ):
        nodes_config = config["nodes"]
        edges_config = config.get("links", {})
        self.context = config["graph"].get("context", [])

        # graph state
        attrs = self._state_attrs(config["graph"], nodes_config)
        self.State = create_dynamic_class("State", attrs)

        workflow = StateGraph(self.State)

        nodes = {}
        edges = {}
        for c in nodes_config:
            node_name = c["label"]
            func = partial(
                self._node,
                node_name=node_name,
                **{
                    k: v
                    for k, v in c.items()
                    if k in {"model", "field", "utter", "jinja", "template", "function"}
                },
            )
            workflow.add_node(node_name, func)
            nodes[node_name] = func

            if "next" in c:
                next_path_map = self._next_path_map(c["next"])
                if "path_map" in c:
                    path_map = c["path_map"]
                else:
                    path_map = next_path_map

                workflow.add_conditional_edges(
                    node_name, partial(self._path, next=c["next"]), path_map
                )
                edges[node_name] = set(next_path_map.keys())

        for edge in edges_config:
            sources = edge["source"]
            if isinstance(edge["source"], Text):
                sources = [edge["source"]]
            for source in sources:
                targets = edges.get(source, set())
                targets.add(edge["target"])
                edges[source] = targets

        for k, v in compiled_nodes.items():
            if k not in edges.keys() and all(
                [k not in targets for targets in edges.values()]
            ):
                continue

            nodes[k] = v
            workflow.add_node(k, v)

        # 如果source有多个节点，必须先将source中的每个节点add_node，然后再add_edge，否则会报错
        for edge in edges_config:
            workflow.add_edge(edge["source"], edge["target"])

        targets = set().union(*edges.values())
        for n in nodes.keys():
            if n not in targets:
                workflow.set_entry_point(n)
            if n not in edges:
                workflow.set_finish_point(n)

        if saver:
            graph = workflow.compile(checkpointer=saver)
        else:
            graph = workflow.compile()

        return graph

    def _node(
        self,
        state: Dict,
        node_name: Text,
        model: Text = None,
        field: Union[Text | List] = None,
        utter: Text = None,
        jinja: Text = None,
        template: Text = None,
        function: Union[Dict, Text] = None,
    ):
        output = None
        if utter:
            if utter == "__EMPTY__":
                output = ""
            else:
                output = utter
        elif template:
            template = render(template, **state)
            if model:
                output = self.chat(template.format(**state), model=model)
            else:
                output = self.chat(template.format(**state))

            vars = re.compile(r"\{(\w+)\}").findall(template)
            vars.append("req_id")
            log_dict = {
                "sample_type": node_name,
                "template": template,
                "output": output,
                **{v: state[v] for v in vars if v in state},
            }
            logger.info(f"[TemplateNode]{json.dumps(log_dict, ensure_ascii=False)}")
        elif jinja:
            output = render(jinja, **state)
        elif function:
            if isinstance(function, Text):
                if function == "default":
                    func_name = node_name
                else:
                    func_name = function
                params = {}
            else:
                func_name = function["name"]
                params = function.get("params", {})

            func = None
            for m in [self.node_module, default_nodes]:
                if not m:
                    continue

                if function_exists(m, func_name):
                    func = getattr(m, func_name)
                    break

            assert func, f"Node function [{func_name}] not found."

            output = func(state, **params)

        if field:
            if isinstance(field, List):
                if isinstance(output, Text):
                    output = json.loads(output)

                if isinstance(output, Dict):
                    output_state = {f: output.get(f) for f in field}
                elif isinstance(output, List):
                    output_state = dict(zip(field, output))
            else:
                output_state = {field: output}
        else:
            output_state = {f"{node_name}_o": output}

        output_state["nodes"] = [node_name]

        logger.info(f"Node - {node_name}: {output_state}")

        return output_state

    def _path(self, state: Dict, next: List[Dict]):

        def process(state: Dict, content: Union[Text, List[Text]]):
            if isinstance(content, List):
                return self._path(state, content)
            else:
                return content

        def if_condition(state: Dict, cond: Dict) -> Optional[Text]:
            if predicate(cond["if"], **state):
                return process(state, cond["then"])
            else:
                return None

        for cond in next[:-1]:
            r = if_condition(state, cond)
            if r:
                return r

        return process(state, next[-1]["else"])

    def _next_path_map(self, next_config: List[Dict]) -> Dict[Text, Text]:

        def process(content: Union[Text, List[Dict]]) -> Dict[Text, Text]:
            path_map: Dict[Text, Text] = {}
            if isinstance(content, List):
                path_map.update(self._next_path_map(content))
            else:
                path_map[content] = content

            return path_map

        path_map: Dict[Text, Text] = {}
        for n in next_config[:-1]:
            path_map.update(process(n["then"]))

        path_map.update(process(next_config[-1]["else"]))

        return path_map

    def _state_attrs(self, graph_config: Dict, nodes_config: Dict) -> Dict:
        attrs = {
            s: self._create_type(t)
            for s, t in graph_config.get("graph_default", {}).items()
        }
        for c in nodes_config:
            if c.get("field"):
                if isinstance(c["field"], Text):
                    attrs[c["field"]] = Text
                elif isinstance(c["field"], List):
                    for f in c["field"]:
                        attrs[f] = Text
                elif isinstance(c["field"], Dict):
                    for k, v in c["field"].items():
                        attrs[k] = self._create_type(v)
            else:
                attrs[f"{c['label']}_o"] = Text

        attrs["messages"] = Annotated[List[AnyMessage], add_messages]
        attrs["nodes"] = Annotated[List[Text], add]
        return attrs

    def _create_type(self, type_name: Text):
        if type_name in self.type_dict:
            return self.type_dict[type_name]
        else:
            raise ValueError(f"Unknown type name: {type_name}")

    def draw(self, path: Text):
        assert path
        self.graph.get_graph(xray=1).draw_mermaid_png(output_file_path=path)

    def run(self, debug=False, **kwargs):
        res = self.graph.invoke(
            input=self.State(**kwargs),
            config={"configurable": {"thread_id": "1"}},
            debug=debug,
        )
        logger.info(colored(f"Executed Nodes: {res.get('nodes')}", "cyan"))
        return res
