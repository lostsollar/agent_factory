import logging
import os
import yaml

from enum import Enum
from typing import Text
from types import ModuleType

from . import nodes
from ...src import Graph


logger = logging.getLogger(__name__)


class AgentsEnum(Enum):
    EXTRACTOR = "extractor"
    GENERATOR = "generator"


class AgentsManager:
    def __init__(self, type_dict={}):
        self._agents = {
            x.value: self.create_agent(f"{x.value}.yaml", nodes, type_dict)
            for x in AgentsEnum
        }

    def create_agent(self, config_path: Text, nodes: ModuleType = None, type_dict={}):
        dir = os.path.dirname(__file__)
        config_path = os.path.join(dir, config_path)

        with open(config_path, "r") as f:
            graph_config = yaml.safe_load(f)

        graph = Graph(
            graph_config,
            nodes,
            type_dict=type_dict,
            saver=None,
        )
        # graph.draw(config_path.replace(".yaml", ".png"))

        return graph

    def get_agent(self, agent_name: AgentsEnum):
        return self._agents[agent_name.value]


if __name__ == "__main__":
    agents_manager = AgentsManager()
    slots_extractor = agents_manager.get_agent(AgentsEnum.EXTRACTOR)
    generator = agents_manager.get_agent(AgentsEnum.GENERATOR)

    history = ["Assistant: 吃了么您呐"]
    while True:
        print(f"================== History ==================")
        [print(u) for u in history]
        query = input("User: ")
        history.append(f"User: {query}")

        slots_state = slots_extractor.run(query=query, history=history)
        # slots_dict = {
        #     x: slots_state[f"{x}_o"]
        #     for x in slots_state["nodes"]
        #     if f"{x}_o" in slots_state
        # }
        # slots_dict.update(slots_state)
        slots_dict = slots_state
        logger.info(f"slots: {slots_dict}")

        state = generator.run(query=query, history=history, slots=slots_dict)
        response = state["response"]
        logger.info(f"response: {response}")

        history.append(f"Assistant: {response}")
