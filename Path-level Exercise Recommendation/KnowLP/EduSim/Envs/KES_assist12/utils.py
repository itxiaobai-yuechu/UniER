

import os
from longling import json_load, path_append, abs_current_dir
from EduSim.Envs.shared.KSS_KES import KS
from EduSim.utils.io_lib import load_ks_from_csv



def load_configuration(filepath):
    return json_load(filepath)

def load_knowledge_structure(filepath):
    knowledge_structure = KS()
    knowledge_structure.add_edges_from([list(map(int, edges)) for edges in load_ks_from_csv(filepath)])
    return knowledge_structure


def load_learning_order(filepath):
    return json_load(filepath)

def load_items(filepath):
    if os.path.exists(filepath):
        return json_load(filepath)
    else:
        return {}

def load_concept_difficulty(filepath):
    if os.path.exists(filepath):
        return json_load(filepath)
    else:
        return {}

def load_knowledge_structure_sim(filepath):
    knowledge_structure_sim = KS()
    knowledge_structure_sim.add_edges_from([list(map(int, edges)) for edges in load_ks_from_csv(filepath)])
    return knowledge_structure_sim

def load_environment_parameters(directory=None):
    if directory is None:
        directory = os.path.abspath(os.path.join(abs_current_dir(__file__), "../../../data/dataProcess/assist12"))
    return {
        "configuration": load_configuration(path_append(directory, "configuration.json")),
        "knowledge_structure": load_knowledge_structure(path_append(directory, "knowledge_structure.csv")),
        "learning_order": load_learning_order(path_append(directory, "learning_order.json")),
        "items": load_items(path_append(directory, "items.json")),
        "know_item": load_items(path_append(directory, "know_item.json")),
        "concept_difficulty": load_concept_difficulty(path_append(directory, "concept_difficulty.json")),
        "knowledge_structure_sim": load_knowledge_structure_sim(path_append(directory, "knowledge_structure_sim.csv"))
    }