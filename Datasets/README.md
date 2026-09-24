# Unified datasets

Create one directory per normalized dataset name:

```text
Datasets/
  assist2017/
    train_valid.csv
    test.csv
    train_valid_sequences.csv
    test_sequences.csv
    pkm.pth
    pkc.pth
    graph_vertex.json
    dataOff
    dataRec
    student_log_kt_None
    MyTransitionGraph.npy
    nxgraph.pkl
    prerequisite.json
    assist17GraphEmbedding.ckpt
    ASSIST17GraphEmbedding.npy
    models/
      ac/
        env_weights/ValBest.ckpt
        env_weights/ValBest.pt
        agent_weights/ValBest.ckpt
        agent_weights/ValBest.pt
      dkt/env_weights/ValBest.ckpt
      akt/qid_test_predictions.txt
      simplekt/qid_test_predictions.txt
      muloer-san/qid_test_question_predictions.txt
```

The files above are a superset. You only need the subset required by the model you run.

| Model | Canonical inputs |
| --- | --- |
| DRE | `train_valid.csv`, `test.csv`, `pkm.pth` |
| AKT | `test_sequences.csv`, `models/akt/qid_test_predictions.txt` |
| SimpleKT | `test_sequences.csv`, `models/simplekt/qid_test_predictions.txt` |
| MMER | `train_valid.csv`, `test.csv`, `pkm.pth` |
| KCP-ER | `test_sequences.csv`, `pkm.pth`, `pkc.pth` |
| MulOER-SAN | `test.csv`, `models/muloer-san/qid_test_question_predictions.txt` |
| KG4EX | `train_valid_sequences.csv`, `test_sequences.csv`, `pkm.pth`, `pkc.pth` |
| ER-TGA | `train_valid.csv`, `test.csv`, `test_sequences.csv`, `pkm.pth` |
| NR4DER | `train_valid_sequences.csv`, `test_sequences.csv` |
| Item-level DKT/Ep evaluation | `models/dkt/env_weights/ValBest.ckpt` |
| Every path-level model | Common PLER CSV, split logs, transition graph, prerequisite graph, and graph embeddings shown above |
| AC | Common PLER artifacts plus `models/ac/{env_weights,agent_weights}/ValBest.{ckpt,pt}` |

Run commands from the repository root. The CLI creates the original model-specific directory layout automatically; do not manually copy these inputs into individual model folders.
