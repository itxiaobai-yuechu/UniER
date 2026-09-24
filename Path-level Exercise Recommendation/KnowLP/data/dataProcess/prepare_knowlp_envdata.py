import os
import json
import csv
import pandas as pd
from collections import defaultdict

def generate_configuration(config_data, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=4)

def generate_knowledge_structure(prerequisite_path, output_path):
    if not os.path.exists(prerequisite_path):
        raise FileNotFoundError(prerequisite_path)

    with open(prerequisite_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    edges = []
    if isinstance(data, list):
        for pair in data:
            if len(pair) == 2:
                pre, post = int(pair[0]), int(pair[1])
                edges.append((post, pre))
    elif isinstance(data, dict):
        for pre, posts in data.items():
            for post in posts:
                edges.append((int(post), int(pre)))

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for u, v in sorted(set(edges)):
            writer.writerow([u, v])


def generate_learning_order(prerequisite_path, train_path, test_path, output_path):
    if not os.path.exists(prerequisite_path):
        return

    with open(prerequisite_path, 'r', encoding='utf-8') as f:
        edges = json.load(f)

    adj = defaultdict(list)
    indegree = {}
    nodes_in_edges = set()

    for u, v in edges:
        adj[u].append(v)
        if u not in indegree: indegree[u] = 0
        if v not in indegree: indegree[v] = 0
        indegree[v] += 1
        nodes_in_edges.add(u)
        nodes_in_edges.add(v)

    from collections import deque
    queue = deque([n for n in nodes_in_edges if indegree[n] == 0])
    order = []

    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in adj[node]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    df_all = pd.concat([df_train, df_test], ignore_index=True)
    
    all_concepts = set()
    for _, row in df_all.iterrows():
        cs = str(row['concepts']).split(',')
        for c in cs:
            c = c.strip()
            if c:
                all_concepts.add(int(c))

    remaining_nodes = sorted(list(all_concepts - set(order)))

    final_order = order + remaining_nodes

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_order, f, indent=4)
        

def generate_items(train_path, test_path, output_path):
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    df_all = pd.concat([df_train, df_test], ignore_index=True)

    q_knowledge_map = dict()
    q_stats = defaultdict(lambda: {"total": 0, "wrong": 0})

    for _, row in df_all.iterrows():
        qs = str(row['questions']).split(',')
        cs = str(row['concepts']).split(',')
        rs = str(row['responses']).split(',')
        
        if len(qs) == len(cs) == len(rs):
            for q, c, r in zip(qs, cs, rs):
                q, c, r = q.strip(), c.strip(), r.strip()
                if not (q and c and r):
                    continue

                qid = int(q)
                cid = int(c)
                q_stats[qid]["total"] += 1
                if int(r) == 0:
                    q_stats[qid]["wrong"] += 1

                if qid not in q_knowledge_map:
                    q_knowledge_map[qid] = []
                if cid not in q_knowledge_map[qid]:
                    q_knowledge_map[qid].append(cid)

    know_count = defaultdict(int)
    for qid, kns in q_knowledge_map.items():
        for k in kns:
            know_count[k] += 1

    group_pointers = defaultdict(int)
    q_final_assign = {}

    for qid in sorted(q_knowledge_map.keys()):
        kns = q_knowledge_map[qid]
        key = tuple(kns)

        idx = group_pointers[key] % len(kns)
        choose = kns[idx]
        q_final_assign[qid] = choose
        group_pointers[key] += 1

    for qid, kns in q_knowledge_map.items():
        for k in kns:
            if know_count[k] == 1:
                q_final_assign[qid] = k
                break

    final_items = {}
    for qid in sorted(q_final_assign.keys()):
        knowledge = q_final_assign[qid]
        stats = q_stats[qid]
        diff = stats["wrong"] / stats["total"] if stats["total"] else 0.5
        final_items[str(qid)] = {
            "knowledge": knowledge,
            "difficulty": diff
        }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_items, f, indent=4)


def generate_know_item(train_path, test_path, output_path):
    know_item_dict = defaultdict(set)
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    df_all = pd.concat([df_train, df_test], ignore_index=True)

    for _, row in df_all.iterrows():
        qs = str(row['questions']).split(',')
        cs = str(row['concepts']).split(',')
        if len(qs) == len(cs):
            for q, c in zip(qs, cs):
                q, c = q.strip(), c.strip()
                if q and c:
                    know_item_dict[int(c)].add(int(q))

    result_dict = {str(k): sorted(list(v)) for k, v in sorted(know_item_dict.items())}
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, indent=4)
        


def generate_concept_difficulty(train_path, test_path, output_path):
    import pandas as pd
    import json
    from collections import defaultdict
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    df_all = pd.concat([df_train, df_test], ignore_index=True)

    concept_stats = defaultdict(lambda: {"total": 0, "wrong": 0})

    for _, row in df_all.iterrows():
        qs = str(row['questions']).split(',')
        cs = str(row['concepts']).split(',')
        rs = str(row['responses']).split(',')
        if len(qs) == len(cs) == len(rs):
            for q, c, r in zip(qs, cs, rs):
                c = c.strip()
                r = r.strip()
                if c and r:
                    cid = int(c)
                    concept_stats[cid]["total"] += 1
                    if int(r) == 0:
                        concept_stats[cid]["wrong"] += 1

    result = {str(cid): (concept_stats[cid]["wrong"] / concept_stats[cid]["total"] if concept_stats[cid]["total"] else 0.5)
              for cid in sorted(concept_stats.keys())}

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4)


def supplement_knowledge_structure_with_order(knowledge_structure_path, learning_order_path, output_path=None):

    if output_path is None:
        output_path = knowledge_structure_path

    if not os.path.exists(knowledge_structure_path):
        return
    if not os.path.exists(learning_order_path):
        return

    existing_edges = set()
    existing_sources = set()

    with open(knowledge_structure_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2 and row[0].strip() and row[1].strip():
                u, v = int(row[0]), int(row[1])
                existing_edges.add((u, v))
                existing_sources.add(u)

    with open(learning_order_path, 'r', encoding='utf-8') as f:
        order = json.load(f)

    added_edges = set()
    added_sources = set()

    for i in range(1, len(order)):
        prev_k = int(order[i - 1])
        curr_k = int(order[i])

        if curr_k in existing_sources or curr_k in added_sources:
            continue

        edge = (curr_k, prev_k)
        if edge not in existing_edges:
            added_edges.add(edge)
            added_sources.add(curr_k)

    all_edges = sorted(existing_edges | added_edges)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for edge in all_edges:
            writer.writerow(edge)

    
def generate_knowledge_structure_sim(similarity_path, output_path):
    if not os.path.exists(similarity_path):
        raise FileNotFoundError(similarity_path)

    with open(similarity_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    edges = set()

    for item in data:
        if len(item) >= 2:
            i, j = int(item[0]), int(item[1])

            edges.add((i, j))
            edges.add((j, i))

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for u, v in sorted(edges):
            writer.writerow([u, v])


if __name__ == "__main__":
    dataset = os.environ.get("UNIER_PATH_DATASET", "assist17")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), dataset))
    os.makedirs(base_dir, exist_ok=True)

    train_csv = os.path.join(base_dir, "train_valid.csv")
    test_csv = os.path.join(base_dir, "test.csv")
    
    config_dict = {
        "binary_scorer": True,
        "order_ratio": 0.5,
        "review_times": 3,
        "initial_steps": 5,
        "exam_sum": True
    }
    generate_configuration(config_dict, os.path.join(base_dir, "configuration.json"))
    
    knowledge_structure_csv = os.path.join(base_dir, "knowledge_structure.csv")
    learning_order_json = os.path.join(base_dir, "learning_order.json")

    generate_knowledge_structure(
        os.path.join(base_dir, "prerequisite.json"),
        knowledge_structure_csv
    )
    
    generate_learning_order(
        os.path.join(base_dir, "prerequisite.json"),
        train_csv,
        test_csv,
        learning_order_json
    )

    supplement_knowledge_structure_with_order(
        knowledge_structure_csv,
        learning_order_json,
        knowledge_structure_csv
    )
    
    generate_items(train_csv, test_csv, os.path.join(base_dir, "items.json"))
    generate_know_item(train_csv, test_csv, os.path.join(base_dir, "know_item.json"))
    generate_concept_difficulty(
        train_csv, test_csv, os.path.join(base_dir, "concept_difficulty.json")
    )

    generate_knowledge_structure_sim(
        os.path.join(base_dir, "similarity.json"),
        os.path.join(base_dir, "knowledge_structure_sim.csv")
    )
