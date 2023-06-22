import numpy as np

from typing import List

try:
    from typing import Literal, Final, Annotated, TypedDict, Protocol
except Exception:
    from typing_extensions import Literal, Final, Annotated, TypedDict, Protocol


def _infer_demo_ensemble(ensemble_demo_k,
                         sorted_indices,
                         all_demos_raw,
                         all_raw_prompts,
                         all_raw_few_shot_xs,
                         demo_predictions,
                         candidate_topn=100):
    assert candidate_topn >= ensemble_demo_k
    all_demos_raw = np.array(all_demos_raw)
    all_raw_prompts = np.array(all_raw_prompts)
    all_raw_few_shot_xs = np.array(all_raw_few_shot_xs)

    candidates_prompts = all_raw_prompts[sorted_indices][:candidate_topn]
    candidates_demos = all_demos_raw[sorted_indices][:candidate_topn]
    candidates_values = demo_predictions[sorted_indices][:candidate_topn].numpy()
    candidates_raw_few_shot_xs = all_raw_few_shot_xs[sorted_indices][:candidate_topn]

    # normalize candidates_values
    candidates_values = candidates_values / max(candidates_values)

    picked_index = 0
    selected_prompts = [candidates_prompts[picked_index]]
    selected_demos = [candidates_demos[picked_index]]
    selected_values = [candidates_values[picked_index]]
    selected_raw_few_shot_xs = [candidates_raw_few_shot_xs[picked_index]]

    # Correspond to Algorithm 2: Greedy Algorithm
    while len(selected_prompts) < ensemble_demo_k:
        excluded_indices = [picked_index]
        exclude_mask = np.ones(len(candidates_demos), dtype=bool)
        exclude_mask[excluded_indices] = False

        candidates_prompts = candidates_prompts[exclude_mask]
        candidates_demos = candidates_demos[exclude_mask]
        candidates_values = candidates_values[exclude_mask]
        candidates_raw_few_shot_xs = candidates_raw_few_shot_xs[exclude_mask]

        demo_graph = DemoGraph(selected_prompts,
                               selected_demos,
                               selected_values,
                               candidates_prompts,
                               candidates_demos,
                               candidates_values)
        picked_index, node_scores = demo_graph.pick_best_candidate_node()

        selected_prompts.append(candidates_prompts[picked_index])
        selected_demos.append(candidates_demos[picked_index])
        selected_values.append(candidates_values[picked_index])
        selected_raw_few_shot_xs.append(candidates_raw_few_shot_xs[picked_index])

        assert len(selected_prompts) == len(selected_demos) == len(selected_values) == len(
            selected_raw_few_shot_xs)
        assert len(node_scores) == len(candidates_prompts) == len(candidates_demos) == len(
            candidates_values)

    return selected_prompts, selected_demos, selected_values, selected_raw_few_shot_xs


class DemoNode:
    def __init__(self,
                 demo_str_list: np.ndarray,
                 prompt_str: str,
                 value: float):
        self.demo_str_list = tuple([prompt_str] + demo_str_list.tolist())
        self.value = value

    @property
    def demo_set(self):
        return set(self.demo_str_list)

    @property
    def demo_size(self):
        return len(self.demo_str_list)


class DemoGraph:
    def __init__(self,
                 selected_prompts: List[str],
                 selected_demos: List[np.ndarray],
                 selected_values: List[float],
                 candidates_prompts: List[str],
                 candidates_demos: List[np.ndarray],
                 candidates_values: np.ndarray
                 ):

        # create selected nodes
        self.selected_nodes = []
        for demo_str_list, prompt_str, value in zip(selected_demos, selected_prompts, selected_values):
            self.selected_nodes.append(DemoNode(demo_str_list, prompt_str, value))

        # create candidate nodes
        self.candidate_nodes = []
        for demo_str_list, prompt_str, value in zip(candidates_demos, candidates_prompts, candidates_values):
            self.candidate_nodes.append(DemoNode(demo_str_list, prompt_str, value))

    def _compute_node_difference(self,
                                 node1: DemoNode,
                                 node2: DemoNode):
        assert node1.demo_size == node2.demo_size
        return 1 - len(node1.demo_set.intersection(node2.demo_set)) / node1.demo_size

    def pick_best_candidate_node(self):
        node_scores = []
        for node in self.candidate_nodes:
            node_differences = [self._compute_node_difference(node, x) for x in self.selected_nodes]
            node_differences_mean = np.mean(node_differences)
            node_score = 2 * (node.value * node_differences_mean) / (node.value + node_differences_mean)
            node_scores.append(node_score)
        return np.argsort(node_scores)[::-1][0], node_scores
