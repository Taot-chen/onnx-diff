from dataclasses import dataclass
from typing import Any
import onnx.checker
from onnx import ModelProto
from grakel import Graph
from grakel.kernels import ShortestPath
from copy import deepcopy
from utils import print_summary, SummaryResults

@dataclass
class Matches:
    same: int  # Number of items that are exactly the same.
    a_total: int  # Total number of items in A.
    b_total: int  # Total number of items in B.

class OnnxDiff:
    def __init__(self, model_a: ModelProto, model_b: ModelProto):
        self._model_a = model_a
        self._model_b = model_b

    def _onnx_to_edge_list(self, graph):
        nodes = graph.node
        initializer_names = [init.name for init in graph.initializer]
        output_node_hash = {}
        edge_list = []
        for i, node in enumerate(nodes, 0):
            for output in node.output:
                if output in output_node_hash.keys():
                    output_node_hash[output].append(i)
                else:
                    output_node_hash[output] = [i]
        for i, inp in enumerate(graph.input, len(nodes)):
            output_node_hash[inp.name] = [i]
        for i, node in enumerate(nodes, 0):
            for input in node.input:
                if input in output_node_hash.keys():
                    edge_list.extend(
                        [(node_id, i) for node_id in output_node_hash[input]]
                    )
                else:
                    if not input in initializer_names:
                        print("No corresponding output found for {0}.".format(input))
        for i, output in enumerate(graph.output, len(nodes) + len(graph.input) + 1):
            if output.name in output_node_hash.keys():
                edge_list.extend(
                    [(node_id, i) for node_id in output_node_hash[output.name]]
                )
            else:
                pass
        return edge_list

    def _calculate_score(self) -> float:
        a_edges = self._onnx_to_edge_list(self._model_a.graph)
        b_edges = self._onnx_to_edge_list(self._model_b.graph)
        a_graph = Graph(a_edges)
        b_graph = Graph(b_edges)

        sp_kernel = ShortestPath(normalize=True, with_labels=False)

        sp_kernel.fit_transform([a_graph])
        fit = sp_kernel.transform([b_graph])

        return fit[0][0]

    def _safe_remove(self, items: list, x) -> bool:
        for index, item in enumerate(items):
            if type(item) == type(x) and item == x:
                del items[index]
                return True
        return False

    def _match_items(self, a, b) -> Matches:
        a_items = deepcopy(a)
        b_items = deepcopy(b)
        a_total = len(a_items)
        b_total = len(b_items)

        match_count = 0
        while len(a_items) > 0:
            a = a_items.pop()
            if self._safe_remove(b_items, a):
                match_count += 1

        return Matches(same=match_count, a_total=a_total, b_total=b_total)

    def _get_items_from_fields(self, root, ignore_fields=[]):
        items = []
        for field in root.DESCRIPTOR.fields:
            if field.name not in ignore_fields:
                item = getattr(root, field.name)
                items.append(item)
        return items

    def _match_fields(self, a, b, ignore_fields=[]):
        a_items = self._get_items_from_fields(root=a, ignore_fields=ignore_fields)
        b_items = self._get_items_from_fields(root=b, ignore_fields=ignore_fields)
        return self._match_items(a=a_items, b=b_items)

    def _calculate_graph_matches(self) -> dict:
        a_graph = self._model_a.graph
        b_graph = self._model_b.graph
        return {
            "initializers": self._match_items(a_graph.initializer, b_graph.initializer),
            "inputs": self._match_items(a_graph.input, b_graph.input),
            "outputs": self._match_items(a_graph.output, b_graph.output),
            "nodes": self._match_items(a_graph.node, b_graph.node),
            "misc": self._match_fields(
                a_graph,
                b_graph,
                ignore_fields=["initializer", "input", "output", "node"],
            ),
        }

    def _calculate_root_matches(self) -> dict:
        return {
            "misc": self._match_fields(
                self._model_a,
                self._model_b,
                ignore_fields=["graph"],
            ),
        }

    def _validate(self, model: ModelProto) -> bool:
        try:
            onnx.checker.check_model(model)
            return True
        except:
            return False

    def summary(self, output=False) -> SummaryResults:
        results = SummaryResults(
            exact_match=(self._model_a == self._model_b),
            score=self._calculate_score(),
            a_valid=self._validate(self._model_a),
            b_valid=self._validate(self._model_b),
            graph_matches=self._calculate_graph_matches(),
            root_matches=self._calculate_root_matches(),
        )
        if output:
            print_summary(results)
        return results
