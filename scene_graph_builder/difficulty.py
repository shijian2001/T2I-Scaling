import networkx as nx
import numpy as np
import time
from typing import List, Tuple, Dict


def construct_scene_graph(
        objects: List[str], attributes: Dict[str, List[str]], relations: List[Tuple[str, str, str]]
) -> nx.Graph:
    G = nx.Graph()

    for obj in objects:
        G.add_node(obj, type="object")

    for obj, attrs in attributes.items():
        for attr in attrs:
            attr_node = f"{attr}_{obj}"
            G.add_node(attr_node, type="attribute")
            G.add_edge(obj, attr_node)

    for subj, rel, obj in relations:
        rel_node = f"{rel}_{subj}_{obj}"
        G.add_node(rel_node, type="relation")
        G.add_edge(subj, rel_node)
        G.add_edge(rel_node, obj)

    return G


def print_graph_info(G: nx.Graph):
    print("Nodes:")
    for node in G.nodes(data=True):
        print(node)

    print("\nEdges:")
    for edge in G.edges(data=True):
        print(edge)


class SceneGraphDifficulty:
    """Class for calculating the difficulty of scene graphs based on information flow."""

    def __init__(self):
        pass

    def calculate_subgraph_difficulty(self, G: nx.Graph) -> float:

        n_obj = len([node for node, data in G.nodes(data=True) if data.get('type') == 'object'])
        n_att = len([node for node, data in G.nodes(data=True) if data.get('type') == 'attribute'])
        n_rel = len([node for node, data in G.nodes(data=True) if data.get('type') == 'relation'])

        if n_obj == 0:
            return 0.0

        difficulty = n_obj * max(1, n_att / n_obj) * max(1, n_rel / n_obj)

        return float(difficulty)

    def calculate_difficulty(self, G: nx.Graph) -> float:
        if G.number_of_nodes() == 0:
            return 0.0

        subgraphs = [G.subgraph(c) for c in nx.connected_components(G)]

        if not subgraphs:
            return 0.0

        difficulties = [self.calculate_subgraph_difficulty(sg) for sg in subgraphs]
        return sum(difficulties)

