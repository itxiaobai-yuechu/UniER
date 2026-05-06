# UniER
UniER consists of two main types of models: **Item-level Exercise Recommendation** and **Path-level Exercise Recommendation**. Below is a summary of the individual model names and their corresponding execution commands.

---
## Environment Setup
```bash
conda create -n unirec python=3.10
conda activate unirec
pip install -r requirements.txt
```
## Data Preprocessing
Navigate to `pykt-toolkit/examples` and process the dataset to obtain `train_valid.csv`, `test.csv`, `train_valid_sequences.csv`, and `test_sequences.csv`:
```bash
python data_preprocess.py
```
Following the instructions in Data Process, obtain `pkm.pth`, `pkc.pth`, and `qid_test_questions.txt` for subsequent exercise recommendation processing.

##  Item-level Exercise Recommendation
### 1. DRE
**Related execution commands:**
```bash
python data_process.py
python Q.py
python run_this.py

python evaluate_acc.py
python evaluate_ndcg.py

python get_final_pro_rec.py
python merge_data.py
python filter_rec.py
python test_ep_all/portion.py
```

### 2. KT_Base_Rec
First, obtain `qid_test_questions.txt`.
**Related execution commands:**
```bash
# Based on pykt, related recommendation, and testing codes:
python recommend_ex.py
python recommend_ex_list.py
python test_ep_all/portion.py
```

### 3. MMER
**Related execution commands:**
```bash
python to_json.py

python main.py --file_dir=datasets/assist2009 --cpt_num=123

python test_ep_all_MMER.py
python test_ep_por_MMER.py
```

### 4. KCP-ER
**Related execution commands:**
```bash
python select_test300.py
python eb_filter.py
python REL_generator.py
python evaluate4acc_etc.py


python get_rec_20.py
python test_ep_all/portion.py
```

### 5. MulOER-SAN
**Related execution commands:**
```bash
# Preprocessing, training, and predicting based on PYKT:
python ./data_preprocess.py --dataset_name assist2009
python ./wandb_akt_train.py --use_wandb 0 --add_uuid 0 --dataset_name assist2009
python ./wandb_predict.py --use_wandb 0 --save_dir saved_model/assist2009_akt_qid_saved_model

# Obtaining coverage:
python ./transformer.py

# Further filtering and recommending exercises:
python ./chuli.py
python ./rec.py

python fileter_over4_test.py
python get_original_ex_ques.py
python test_ep_all/portion.py
```

### 6. KG4EX
**Related execution commands:**
```bash
# Training, evaluation, and recommendation testing:
python run.py --do_train --cuda --data_path ../data/algebra2005 --model TransE -b 1024 -d 1000 -g 12.0 -a 1.0 -lr 0.001 -adv -save models/algebra2005/TransE_adv
python test_TransE.py

python get_result_20.py
python get_unique_last.py
python test_ep_all/portion.py
```

### 7. ER-TGA
**Related execution commands:**
```bash
./Auto_Run_Scripts.sh
python test_ep_all/portion.py
```

### 8. melt-lstm
**Related execution commands:**
```bash
# After obtaining the Q matrix, proceed with multi-stage processing sequentially:
python Q.py
python stage1_main.py 
python EB_filter.py 
python stage2_main.py 

python evaluate.py
python test_ep_all/portion.py
```



<br>

---

##  Path-level Exercise Recommendation


### 1. SRC
**Related execution commands:**
```bash
# Data processing and DKT acquisition:
python data/dataproces_npz.py
python trainDKT.py -d assist17 -m DKT

# Run SRC script:
python trainSRC.py -d assist17 -p 3 --steps 10 --target_type all -c 0
```

### 2. AC
**Related execution commands:**
```bash
python data/dataProcess/data_process.py
python data/dataProcess/envDKT.py

cd scripts
python runSim.py -s KESassist17 --cudaDevice cuda:0 --max_steps 10 --max_episode_num 10000 --target_type all
```

### 3. DQN
**Related execution commands:**
```bash
python data/dataProcess/data_process.py
python data/dataProcess/envDKT.py

cd scripts
python runSim.py -s KESassist17 --max_steps 10 --target_type all --cudaDevice cuda:0
```

### 4. RLTutor
**Related execution commands:**
```bash
python data/dataProcess/data_process.py
python data/dataProcess/envDKT.py

# Getting weights:
python EduSim/agents/TutoInnerModel.py

cd scripts
python runSim.py -s KESassist17 --max_steps 10 --target_type all
```

### 5. CSEAL
**Related execution commands:**
```bash
python data/dataProcess/data_process.py
python data/dataProcess/BuildTransitionGraph.py
python data/dataProcess/envDKT.py

cd scripts
python runSim.py -s KESassist17 --max_steps 10 --cudaDevice cuda:0 --max_episode_num 10000 --target_type all
```

### 6. GEHRL
**Related execution commands:**
```bash
python data/dataProcess/data_process.py
python data/dataProcess/BuildTransitionGraph.py
python data/dataProcess/envDKT.py
python data/dataProcess/GraphEmbedding.py

cd scripts
python runSim.py -s KESassist17 -m 10 --max_episode_num 10000 --cuda_device 1 --target_type all
```

### 7. DLPR
**Related execution commands:**
```bash
# From pykt/examples
python wandb_dimkt_train.py --dataset_name assist2009 --emb_size 128 --difficult_levels 50 --learning_rate 0.002

python data/dataProcess/data_process.py
python data/dataProcess/BuildTransitionGraph.py
python data/dataProcess/envDKT.py
python data/dataProcess/GraphEmbedding.py
python data/dataProcess/prepare_dlpr_envdata.py

cd IDALPR
python DLPR.py
```

### 8. PKSD
**Related execution commands:**
```bash
python data/dataProcess/data_process.py
python data/dataProcess/BuildTransitionGraph.py
python data/dataProcess/envDKT.py
python data/dataProcess/GraphEmbedding.py

cd scripts
python runSim.py -s KESassist17 --cudaDevice cuda:0 --max_steps 10 --max_episode_num 10000 --target_type all
```

### 9. KnowLP
**Related execution commands:**
```bash
# From pykt/examples
python wandb_dimkt_train.py --dataset_name assist2009 --emb_size 128 --difficult_levels 50 --learning_rate 0.002

# Generate data based on EDU-graphRAG
python EDU-graphRAG/generate_concepts_textgrad.py
python -m graphrag.index --root EDU-graphRAG/ragtest
python EDU-graphRAG/extract_json.py

python data/dataProcess/data_process.py
python data/dataProcess/BuildTransitionGraph.py
python data/dataProcess/envDKT.py
python data/dataProcess/GraphEmbedding.py
python data/dataProcess/prepare_knowlp_envdata.py

python KnowLP.py
```
