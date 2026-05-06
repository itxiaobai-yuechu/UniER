import os
import sys
import warnings
import pickle
import json
import numpy as np
import networkx as nx
from tqdm import tqdm
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)
from EduSim.utils import get_proj_path, get_raw_data_path
cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append('../..')


warnings.filterwarnings('ignore')

def build_graph(base_data_path, graph_type, concept_num, KT_graph_path=None, model_type='GKT'):
    question_list = []
    answer_list = []
    seq_len_list = []
    session_num = 0

    with open(base_data_path, 'r', encoding="utf-8") as f:
        datatxt = f.readlines()
    for index in tqdm(range(len(datatxt)), 'Graph Constructing'):
        line = datatxt[index]
        one_session_data = json.loads(line)
        if len({log[0] for log in one_session_data}) < 10:
            continue
        question_list.append([el[0] for el in one_session_data])
        answer_list.append([el[1] for el in one_session_data])
        seq_len_list.append(len(one_session_data))
        session_num += 1

    graph = None
    if model_type == 'GKT':
        if graph_type == 'Dense':
            graph = build_dense_graph(concept_num)
        elif graph_type == 'Transition':
            graph = build_transition_graph(question_list, seq_len_list, session_num, concept_num)
        elif graph_type == 'KT':
            graph = build_KT_graph(KT_graph_path, concept_num)

    print(graph)
    np.save(f'{get_raw_data_path()}/dataProcess/{dataset}/MyTransitionGraph.npy',graph)


def build_transition_graph(question_list, seq_len_list, session_num, concept_num):
    graph = np.zeros((concept_num, concept_num))
    for i in tqdm(range(session_num), 'trainsition constructing'):
        questions = question_list[i]
        seq_len = seq_len_list[i]
        for j in range(seq_len - 1):
            pre = questions[j]
            next_my = questions[j + 1]
            graph[pre, next_my] += 1
    np.fill_diagonal(graph, 0)
    rowsum = np.array(graph.sum(1))

    def inv(x):
        if x == 0:
            return x
        return 1. / x

    inv_func = np.vectorize(inv)
    r_inv = inv_func(rowsum).flatten()
    r_mat_inv = np.diag(r_inv)
    graph = r_mat_inv.dot(graph)
    return graph


def build_KT_graph(file_path, concept_num):
    graph = np.loadtxt(file_path)
    assert graph.shape[0] == concept_num and graph.shape[1] == concept_num
    return graph


def build_dense_graph(node_num):
    graph = 1. / (node_num - 1) * np.ones((node_num, node_num))
    np.fill_diagonal(graph, 0)
    return graph


if __name__ == '__main__':
    dataset = 'mooccubex'
    if dataset == 'assist09':
        num_skills = 123
    elif dataset == 'assist12':
        num_skills = 265
    elif dataset == 'assist15':
        num_skills = 100
    elif dataset == 'assist17':
        num_skills = 102
    elif dataset == 'algebra2005':
        num_skills = 265
    elif dataset == 'bridge2006':
        num_skills = 493
    elif dataset == 'ednet':
        num_skills = 188
    elif dataset == 'nips34':
        num_skills = 388
    elif dataset == 'junyi':
        num_skills = 39
    elif dataset == 'poj':
        num_skills = 2748
    elif dataset == 'statics2011':
        num_skills = 1223
    elif dataset == 'mooccubex':
        num_skills = 436

    base_data_path = f'{get_raw_data_path()}/dataProcess/{dataset}/student_log_kt_None'
    build_graph(base_data_path, 'Transition',num_skills)
    

    graph = np.load(f'{get_raw_data_path()}/dataProcess/{dataset}/MyTransitionGraph.npy')

    knowledge_structure = nx.DiGraph()
    bina_graph = np.where(graph > 0.06, 1, 0)
    prerequisite_edges = []
    for i in range(bina_graph.shape[0]):
        for j in range(bina_graph.shape[1]):
            if bina_graph[i, j] == 1:
                if [j, i] in prerequisite_edges:
                    if graph[i, j] > graph[j, i]:
                        prerequisite_edges.append([i, j])
                        prerequisite_edges.remove([j, i])
                    else:
                        continue
                else:
                    prerequisite_edges.append([i, j])
    knowledge_structure.add_nodes_from([i for i in range(num_skills)])
    knowledge_structure.add_edges_from(prerequisite_edges)

    pbar = tqdm(desc='Removing cycles')
    while True:
        try:
            cycle = nx.find_cycle(knowledge_structure)
            pbar.update(1)
        except nx.NetworkXNoCycle:
            break

        min_weight = float('inf')
        min_edge = None

        for u, v in cycle:
            weight = graph[u, v]
            if weight < min_weight:
                min_weight = weight
                min_edge = [u, v]

        if min_edge:
            knowledge_structure.remove_edge(min_edge[0], min_edge[1])
            if min_edge in prerequisite_edges:
                prerequisite_edges.remove(min_edge)
    pbar.close()

    knowledge_structure = nx.DiGraph()
    knowledge_structure.add_nodes_from([i for i in range(num_skills)])
    knowledge_structure.add_edges_from(prerequisite_edges)

    with open(f"{get_proj_path()}/dataProcess/{dataset}/nxgraph.pkl", "wb") as file:
        str_my = pickle.dumps(knowledge_structure)
        file.write(str_my)

    _topo_order = list(nx.topological_sort(knowledge_structure))
    print(_topo_order)
    assert not list(nx.algorithms.simple_cycles(knowledge_structure)), "loop in DiGraph"

    prerequisite_json_path = f"{get_proj_path()}/dataProcess/{dataset}/prerequisite.json"
    with open(prerequisite_json_path, "w", encoding="utf-8") as f:
        json.dump(prerequisite_edges, f, indent=2)
    print(f"prerequisite.json saved to {prerequisite_json_path}")
