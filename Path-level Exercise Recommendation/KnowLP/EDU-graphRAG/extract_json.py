import json
import os
import glob
import pandas as pd
import difflib
import re
import networkx as nx


base_path = "./"
dataset_name = "xes3g5m"
graph_vertex_path = os.path.join(base_path, "data/dataProcess", dataset_name, "graph_vertex.json")
graphrag_out_dir = os.path.join(base_path, f"EDU-graphRAG/ragtest/output/{dataset_name}")

def normalize(text):
    text = str(text).lower()
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = text.replace("the ", "")
    text = text.strip()
    return text

with open(graph_vertex_path, 'r', encoding='utf-8') as f:
    vertex_dict = json.load(f)

name_to_id = {normalize(k): v for k, v in vertex_dict.items()}

def get_matched_id(entity_name):
    clean_name = normalize(entity_name)

    if clean_name in name_to_id:
        return name_to_id[clean_name]

    matches = difflib.get_close_matches(clean_name, name_to_id.keys(), n=3, cutoff=0.5)
    if matches:
        return name_to_id[matches[0]]

    return None

relations_file = os.path.join(graphrag_out_dir, "artifacts", "create_final_relationships.parquet")

if not os.path.exists(relations_file):
    exit()

df = pd.read_parquet(relations_file)

prerequisites = set()
similarities = set()

dropped_count = 0
self_loop_count = 0

for _, row in df.iterrows():
    desc = str(row['description']).lower()

    source_id = get_matched_id(row['source'])
    target_id = get_matched_id(row['target'])

    if source_id is None or target_id is None:
        dropped_count += 1
        continue

    if source_id == target_id:
        self_loop_count += 1
        continue

    edge = (source_id, target_id)



    if any(k in desc for k in ["prerequisite", "require", "depend", "foundation"]):
        prerequisites.add(edge)
    elif "similarity" in desc:
        similarities.add(edge)

prerequisites = list(prerequisites)
similarities = list(similarities)

G = nx.DiGraph()

prerequisites = sorted(prerequisites)

clean_edges = []

for u, v in prerequisites:
    G.add_edge(u, v)
    if not nx.is_directed_acyclic_graph(G):
        G.remove_edge(u, v)
    else:
        clean_edges.append((u, v))

prerequisites = clean_edges

out_sim = os.path.join(base_path, "data/dataProcess", dataset_name, "similarity.json")
out_pre = os.path.join(base_path, "data/dataProcess", dataset_name, "prerequisite.json")

with open(out_sim, 'w', encoding='utf-8') as f:
    json.dump(similarities, f, indent=4)

with open(out_pre, 'w', encoding='utf-8') as f:
    json.dump(prerequisites, f, indent=4)
