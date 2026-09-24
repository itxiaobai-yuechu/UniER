

import networkx as nx
import copy

def bfs_hop_nodes(graph, pnode, hop, candidates, mode='pre'):
    assert hop >= 0

    candidates.add(pnode)
    if hop == 0:
        return

    if mode == 'pre':
        for node in list(graph.predecessors(pnode)):
            bfs_hop_nodes(
                graph=graph,
                pnode=node,
                hop=hop - 1,
                candidates=candidates,
                mode='pre'
            )
    else:
        for node in list(graph.successors(pnode)):
            bfs_hop_nodes(
                graph=graph,
                pnode=node,
                hop=hop - 1,
                candidates=candidates,
                mode='back'
            )

def get_goal_neighbors(graph, targets):
    if targets is not None and isinstance(targets, set):
        targets = list(targets)
    
    assert targets is None or isinstance(targets, list), targets

    candidates = set(copy.deepcopy(targets))

    for target_node in targets:
        bfs_hop_nodes(graph, target_node, hop=1, candidates=candidates, mode='pre')

    candidates = list(candidates)

    if not candidates:
        candidates = list(graph.nodes)
    return candidates