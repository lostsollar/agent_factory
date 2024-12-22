from abc import abstractmethod
from enum import Enum
import json
import logging
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, Text, Union
from pydantic import BaseModel

from .utils import render, chat, function_exists, extract_json_object


logger = logging.getLogger(__name__)


class NodeType(Enum):
    UTTER = "utter"
    JINJA = "jinja"
    PROMPT = "prompt"
    FUNCTION = "function"


class PromptNodeConfig(BaseModel):
    model: Text = "gpt-4o"
    enable_json_format: bool = False
    template: Text


class FunctionNodeConfig(BaseModel):
    name: Text
    params: Dict[Text, Text] = {}


class NodeConfig(BaseModel):
    label: Text
    field: Optional[Union[Text | List]]
    type: NodeType
    utter: Optional[Text]
    jinja: Optional[Text]
    prompt: Optional[PromptNodeConfig]
    function: Optional[Union[Text | FunctionNodeConfig]]
    next: Optional[List[Dict[Text, Any]]]


class Node(BaseModel):
    type: NodeType
    config: NodeConfig

    @classmethod
    def create(cls, config: NodeConfig, node_modules: List[ModuleType]) -> "Node":
        if config.type == NodeType.UTTER:
            return UtterNode(config=config)
        elif config.type == NodeType.JINJA:
            return JinjaNode(config=config)
        elif config.type == NodeType.PROMPT:
            return PromptNode(config=config)
        elif config.type == NodeType.FUNCTION:
            if isinstance(config.function, Text):
                if config.function == "default":
                    func_name = config.label
                else:
                    func_name = config.function
                params = {}
            else:
                func_name = config.function.name
                params = config.function.params

            func = None
            for m in node_modules:
                if not m:
                    continue

                if function_exists(m, func_name):
                    func = getattr(m, func_name)
                    break

            return FunctionNode(config=config, func=func, params=params)
        else:
            raise ValueError("Invalid Node Type")

    @property
    def name(self) -> Text:
        return self.config.label

    @abstractmethod
    def run(self, state: Dict) -> Any: ...

    def __call__(self, state: Dict):
        output = self.run(state)

        field = self.config.field
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
            output_state = {f"{self.name}_o": output}

        output_state["nodes"] = [self.name]
        logger.info(f"Node - {self.name}: {output_state}")

        return output_state


class UtterNode(Node):
    type: NodeType = NodeType.UTTER

    def run(self, state: Dict) -> Any:
        if self.utter == "__EMPTY__":
            output = ""
        else:
            output = self.utter
        return output

    @property
    def utter(self):
        return self.config.utter


class JinjaNode(Node):
    type: NodeType = NodeType.JINJA

    def run(self, state: Dict) -> Any:
        return render(self.jinja)

    @property
    def jinja(self):
        return self.config.jinja


class PromptNode(Node):
    type: NodeType = NodeType.PROMPT

    def run(self, state: Dict) -> Any:
        config = self.config.prompt
        template = render(config.template, **state)
        # template = template.format(**state)

        output = chat(template, config.model, config.enable_json_format)
        if config.enable_json_format:
            output = extract_json_object(output)

        return output


class FunctionNode(Node):
    type: NodeType = NodeType.FUNCTION
    func: Callable
    params: Dict = {}

    def run(self, state: Dict) -> Any:
        # 使用 state 中的 value 填充 params 对应的参数值
        for k, v in self.params.items():
            if v in state:
                self.params[k] = state[v]

        return self.func(state, **self.params)


def get_state_value(graph_state: Dict, state_name: Text):
    return graph_state[state_name]


def echo(graph_state: Dict, value):
    return value
