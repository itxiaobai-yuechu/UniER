import argparse
import json
import os
import glob
import pandas as pd
import difflib
import re
import networkx as nx


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BASE_PATH = os.path.dirname(SCRIPT_DIR)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert the latest KnowLP GraphRAG relationships into graph JSON files."
    )
    parser.add_argument("--dataset", default="assist17")
    parser.add_argument("--base-path", default=DEFAULT_BASE_PATH)
    parser.add_argument(
        "--graphrag-output",
        help="GraphRAG run directory, artifacts directory, or relationships parquet file.",
    )
    return parser.parse_args()


def resolve_relations_file(base_path, graphrag_output=None):
    filename = "create_final_relationships.parquet"
    if graphrag_output:
        candidate = os.path.abspath(graphrag_output)
        if os.path.isfile(candidate):
            return candidate
        direct = os.path.join(candidate, filename)
        nested = os.path.join(candidate, "artifacts", filename)
        for path in (direct, nested):
            if os.path.isfile(path):
                return path
        raise FileNotFoundError(f"GraphRAG relationships file not found under: {candidate}")

    output_root = os.path.join(base_path, "EDU-graphRAG", "ragtest", "output")
    candidates = glob.glob(os.path.join(output_root, "*", "artifacts", filename))
    if not candidates:
        raise FileNotFoundError(
            f"No GraphRAG relationships file found under: {output_root}"
        )
    return max(candidates, key=os.path.getmtime)

def normalize(text):
    text = str(text).lower()
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = text.replace("the ", "")
    text = text.strip()
    return text

def main(args):
    graph_vertex_path = os.path.join(
        args.base_path, "data", "dataProcess", args.dataset, "graph_vertex.json"
    )
    relations_file = resolve_relations_file(args.base_path, args.graphrag_output)

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

    similarities = list(similarities)
    graph = nx.DiGraph()
    clean_edges = []

    for source, target in sorted(prerequisites):
        graph.add_edge(source, target)
        if not nx.is_directed_acyclic_graph(graph):
            graph.remove_edge(source, target)
        else:
            clean_edges.append((source, target))

    output_dir = os.path.join(args.base_path, "data", "dataProcess", args.dataset)
    out_sim = os.path.join(output_dir, "similarity.json")
    out_pre = os.path.join(output_dir, "prerequisite.json")

    with open(out_sim, 'w', encoding='utf-8') as f:
        json.dump(similarities, f, indent=4)

    with open(out_pre, 'w', encoding='utf-8') as f:
        json.dump(clean_edges, f, indent=4)

    print(f"GraphRAG relationships: {relations_file}")
    print(f"Dropped unmatched relationships: {dropped_count}")
    print(f"Dropped self-loops: {self_loop_count}")


if __name__ == "__main__":
    main(parse_args())
