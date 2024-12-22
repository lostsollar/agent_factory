from functools import partial
import os

import yaml
from ...src import Graph, chat, load_from_graphml

dir = os.path.dirname(__file__)

graph_path = os.path.join(dir, "configs/agent.graphml")
image_path = graph_path.replace(".graphml", ".png")

# [TODO] load_from_graph function needs to update to adapt to NodeConfig
config = load_from_graphml(graph_path)

with open(os.path.join(dir, "configs/agent.yaml"), "r") as f:
    graph_config = yaml.safe_load(f)
    graph = Graph(
        graph_config,
        chat_func=partial(chat, model="gpt-4o"),
        saver=None,
    )
    graph.draw(path=image_path)
    graph.run(
        topic="价格问题",
        worries_lst="家庭经济情况无法承受,觉得课程性价比低",
    )
