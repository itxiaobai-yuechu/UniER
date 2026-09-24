import os
import json
import argparse
import pandas as pd
from collections import defaultdict


def _normalize_id(raw_id):
    sid = str(raw_id).strip()
    return sid[:-2] if sid.endswith('.0') else sid

def generate_graph_vertex(dataset_name, raw_data_path, output_dir):
    print(f"Processing {dataset_name}...")
    
    skill_id_col = None
    skill_name_col = None
    sep = ','
    
    if dataset_name == "assist2009":
        skill_id_col = "skill_id"
        skill_name_col = "skill_name"
    elif dataset_name == "assist2012":
        skill_id_col = "skill_id"
        skill_name_col = "skill"
    elif dataset_name == "assist2015":
        skill_id_col = "sequence_id"
        skill_name_col = None 
    elif dataset_name == "assist2017":
        skill_id_col = None
        skill_name_col = "skill"
    elif dataset_name == "algebra2005":
        skill_id_col = None
        skill_name_col = "KC(Default)"
        sep = '\t'
    elif dataset_name == "bridge2algebra2006":
        skill_id_col = None
        skill_name_col = "KC(SubSkills)"
        sep = '\t'
    elif dataset_name == "junyi2015":
        skill_id_col = None
        skill_name_col = "topic"
    elif dataset_name == "nips_task34":
        skill_id_col = "SubjectId_level3_str"
        skill_name_col = None
    elif dataset_name in ["ednet", "ednet5w"]:
        skill_id_col = "tags"
        skill_name_col = None
    elif dataset_name == "poj":
        skill_id_col = "Problem"
        skill_name_col = None
    elif dataset_name == "statics2011":
        skill_id_col = None
        skill_name_col = "KC (F2011)"
    elif dataset_name == "slepemapy":
        skill_id_col = "place_asked"
        skill_name_col = None
        sep = ';'
    elif dataset_name == "mooccubex":
        skill_id_col = None
        skill_name_col = None
    else:
        print(f"Warning: '{dataset_name}' uses default processing, name is consistent with ID.")
    id2name = {}

    if dataset_name == "nips_task34":
        meta_data_dir = os.path.join(os.path.dirname(raw_data_path), "metadata", "subject_metadata.csv")
        if os.path.exists(meta_data_dir):
            try:
                df_meta = pd.read_csv(meta_data_dir, dtype=str)
                for _, row in df_meta.iterrows():
                    s_id = str(row['SubjectId']).strip()
                    s_name = str(row['Name']).strip()
                    if s_id and s_id.lower() not in ['nan', 'na', 'null']:
                        id2name[s_id] = s_name
                print(f"  [+] Successfully parsed {len(id2name)} knowledge concept name mappings from nips_task34/metadata/subject_metadata.csv.")
            except Exception as e:
                print(f"  [!] Error reading nips_task34 metadata: {e}")
            else:
                print(f"  [!] Metadata file for nips_task34 not found: {meta_data_dir}")

    elif skill_id_col and skill_name_col and os.path.exists(raw_data_path):
        try:
            df = pd.read_csv(raw_data_path, usecols=[skill_id_col, skill_name_col], sep=sep, encoding='utf-8', encoding_errors='ignore', dtype=str)
            df = df.dropna(subset=[skill_id_col])
            df = df.drop_duplicates(subset=[skill_id_col])
            
            for _, row in df.iterrows():
                s_id = str(row[skill_id_col]).strip()
                if s_id.endswith('.0'):
                    s_id = s_id[:-2]
                s_name = str(row[skill_name_col]).strip()
                if s_id and s_id.lower() not in ['nan', 'na', 'null']:
                    if not s_name or s_name.lower() in ['nan', 'na', 'null']:
                        s_name = s_id
                    id2name[s_id] = s_name
        except Exception as e:
            print(f"  [!] Error reading {dataset_name} raw CSV: {e}，will use skill ID as name by default.")
    elif raw_data_path and not os.path.exists(raw_data_path):
         print(f"  [!] Dataset file does not exist {raw_data_path}，will use skill ID as name by default.")

    keyid_path = os.path.join(output_dir, "keyid2idx.json")
    if not os.path.exists(keyid_path):
        print(f"  [!] {keyid_path} does not exist. You must generate this file using pykt first!")
        return
        
    with open(keyid_path, 'r', encoding='utf-8') as f:
        keyid_data = json.load(f)
        
    if "concepts" not in keyid_data:
        print(f"  [!] {keyid_path} does not contain 'concepts' key. The dataset may not have knowledge concepts.")
        return
        
    concept_mapping = keyid_data["concepts"]

    concept_entries = []
    for origin_id, new_idx in concept_mapping.items():
        if '_' in origin_id and dataset_name not in ["algebra2005", "bridge2algebra2006"]:
            ids = origin_id.split('_')
            names = []
            for sub_id in ids:
                query_id = _normalize_id(sub_id)
                names.append(id2name.get(query_id, sub_id))
            final_name = "_".join(names)
        else:
            query_id = _normalize_id(origin_id)
            final_name = id2name.get(query_id, origin_id)

        concept_entries.append((origin_id, int(new_idx), final_name))

    name_buckets = defaultdict(list)
    for origin_id, idx, base_name in concept_entries:
        name_buckets[base_name].append((origin_id, idx))

    graph_vertex = {}
    for base_name, items in name_buckets.items():
        items.sort(key=lambda x: x[1])
        if len(items) == 1:
            graph_vertex[base_name] = items[0][1]
        else:
            for i, (_, idx) in enumerate(items, start=1):
                graph_vertex[f"{base_name}_{i}"] = idx
        
    sorted_graph_vertex = dict(sorted(graph_vertex.items(), key=lambda item: item[1]))
        
    output_file = os.path.join(output_dir, "graph_vertex.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sorted_graph_vertex, f, ensure_ascii=False, indent=4)
        
    print(f"  -> Successfully generated {output_file}, containing {len(graph_vertex)} knowledge concepts.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="assist2017", help="Dataset name, e.g., assist2009, assist2012, all for processing all")
    args = parser.parse_args()

    DATASETS = {
        "assist2009": "../data/assist2009/skill_builder_data_corrected_collapsed.csv",
        "assist2012": "../data/assist2012/2012-2013-data-with-predictions-4-final.csv",
        "assist2015": "../data/assist2015/2015_100_skill_builders_main_problems.csv",
        "assist2017": "../data/assist2017/anonymized_full_release_competition_dataset.csv",
        "algebra2005": "../data/algebra2005/algebra_2005_2006_train.txt",
        "bridge2algebra2006": "../data/bridge2algebra2006/bridge_to_algebra_2006_2007_train.txt",
        "statics2011": "../data/statics2011/AllData_student_step_2011F.csv",
        "nips_task34": "../data/nips_task34/train_task_3_4.csv",
        "poj": "../data/poj/poj_log.csv",
        "slepemapy": "../data/slepemapy/answer.csv",
        "junyi2015": "../data/junyi2015/junyi_ProblemLog_original.csv",
        "ednet": "../data/ednet/contents/questions.csv",
        "ednet5w": "../data/ednet5w/contents/questions.csv",
        "mooccubex": "../data/mooccubex/concept-problem.txt",
    }
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.abspath(os.path.join(base_dir, ".."))

    for dname, raw_rel_path in DATASETS.items():
        if args.dataset_name != "all" and dname != args.dataset_name:
            continue
            
        raw_path = os.path.join(workspace_root, raw_rel_path.strip("../"))
        output_directory = os.path.join(workspace_root, "data", dname)
        
        if os.path.isdir(output_directory):
            generate_graph_vertex(dname, raw_path, output_directory)
        else:
            print(f"Skipping {dname}, output directory {output_directory} not found.")

