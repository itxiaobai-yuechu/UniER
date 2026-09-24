from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple


@dataclass(frozen=True)
class CommandSpec:
    label: str
    cwd: str
    argv: Tuple[str, ...]
    description: str = ""
    optional: bool = False


@dataclass(frozen=True)
class ModelSpec:
    name: str
    level: str
    aliases: Tuple[str, ...] = ()
    prepare: Tuple[CommandSpec, ...] = ()
    train: Tuple[CommandSpec, ...] = ()
    evaluate: Tuple[CommandSpec, ...] = ()
    notes: Tuple[str, ...] = ()


PY = "{python}"
ITEM = "Item-level Exercise Recommendation"
PATH = "Path-level Exercise Recommendation"


def cmd(label: str, cwd: str, *argv: str, description: str = "", optional: bool = False) -> CommandSpec:
    return CommandSpec(label, cwd, tuple(argv), description, optional)


def path_prepare(model: str, graph: bool = False, extra: Iterable[str] = ()) -> Tuple[CommandSpec, ...]:
    root = f"{PATH}/{model}"
    commands = [
        cmd("data_process", root, PY, "data/dataProcess/data_process.py"),
        cmd("transition_graph", root, PY, "data/dataProcess/BuildTransitionGraph.py"),
        cmd("env_dkt", root, PY, "data/dataProcess/envDKT.py"),
    ]
    if graph:
        commands.append(cmd("graph_embedding", root, PY, "data/dataProcess/GraphEmbedding.py"))
    for script in extra:
        commands.append(cmd(script.rsplit("/", 1)[-1].removesuffix(".py"), root, PY, script))
    return tuple(commands)


def path_runsim(model: str, device_flag: str = "--cudaDevice") -> Tuple[CommandSpec, ...]:
    root = f"{PATH}/{model}/scripts"
    args = [
        PY,
        "runSim.py",
        "-s",
        "{simulator}",
        device_flag,
        "{device_value}",
        "--max_steps",
        "{max_steps}",
        "--max_episode_num",
        "{episodes}",
        "--target_type",
        "{target_type}",
        "--seed",
        "{seed}",
    ]
    return (cmd("train_path_model", root, *args),)


MODEL_SPECS: Tuple[ModelSpec, ...] = (
    ModelSpec(
        "DRE",
        "item",
        aliases=("dre",),
        prepare=(
            cmd("data_process", f"{ITEM}/DRE/data", PY, "data_process.py"),
            cmd("build_q_matrix", f"{ITEM}/DRE", PY, "Q.py"),
        ),
        train=(cmd("train_dre", f"{ITEM}/DRE", PY, "run_this_GPU.py"),),
        evaluate=(
            cmd("accuracy_metrics", f"{ITEM}/DRE", PY, "evaluate_acc.py"),
            cmd("ranking_metrics", f"{ITEM}/DRE", PY, "evaluate_ndcg.py"),
            cmd("dre_postprocess", ".", PY, "-m", "Unier_cli.postprocess", "dre", "--dataset", "{dataset}"),
            cmd("proximity", f"{ITEM}/DRE", PY, "evaluate_proximity.py"),
            cmd("ep_all", ITEM, PY, "test_ep_all.py"),
            cmd("ep_portion", ITEM, PY, "test_ep_por.py"),
        ),
    ),
    ModelSpec(
        "AKT",
        "item",
        aliases=("akt",),
        train=(
            cmd("recommend", f"{ITEM}/KT_Base_Rec", PY, "recommend_ex.py"),
            cmd("recommend_list", f"{ITEM}/KT_Base_Rec", PY, "recommend_ex_list.py"),
        ),
        evaluate=(
            cmd("filter_test", f"{ITEM}/KT_Base_Rec/Ep", PY, "filter_over1_test.py"),
            cmd("ep_all", ITEM, PY, "test_ep_all.py"),
            cmd("ep_portion", ITEM, PY, "test_ep_por.py"),
        ),
    ),
    ModelSpec(
        "SimpleKT",
        "item",
        aliases=("simplekt", "simple-kt"),
        train=(
            cmd("recommend", f"{ITEM}/KT_Base_Rec", PY, "recommend_ex.py"),
            cmd("recommend_list", f"{ITEM}/KT_Base_Rec", PY, "recommend_ex_list.py"),
        ),
        evaluate=(
            cmd("filter_test", f"{ITEM}/KT_Base_Rec/Ep", PY, "filter_over1_test.py"),
            cmd("ep_all", ITEM, PY, "test_ep_all.py"),
            cmd("ep_portion", ITEM, PY, "test_ep_por.py"),
        ),
    ),
    ModelSpec(
        "MMER",
        "item",
        aliases=("mmer",),
        prepare=(
            cmd("to_json", f"{ITEM}/MMER", PY, "to_json.py"),
            cmd("build_q_matrix", f"{ITEM}/MMER", PY, "Q.py"),
        ),
        train=(
            cmd(
                "train_mmer",
                f"{ITEM}/MMER",
                PY,
                "main.py",
                "--file_dir",
                "datasets/{dataset}",
                "--cpt_num",
                "{concept_count}",
                "--epoch",
                "{epochs}",
                "--batch_size",
                "{batch_size}",
                "--cuda",
                "{device}",
                "--seed",
                "{seed}",
            ),
        ),
        evaluate=(
            cmd("accuracy_metrics", f"{ITEM}/MMER", PY, "evaluate_acc.py"),
            cmd("ranking_metrics", f"{ITEM}/MMER", PY, "evaluate_ndcg.py"),
            cmd("ep_all", f"{ITEM}/MMER", PY, "test_ep_all_MMER.py"),
            cmd("ep_portion", f"{ITEM}/MMER", PY, "test_ep_por_MMER.py"),
        ),
    ),
    ModelSpec(
        "KCP-ER",
        "item",
        aliases=("kcper", "kcp_er", "kcp-er"),
        prepare=(cmd("select_test", f"{ITEM}/kcp_er", PY, "select_test300.py", "--dataset", "{dataset}", "--seed", "{seed}"),),
        train=(
            cmd("eb_filter", f"{ITEM}/kcp_er", PY, "eb_filter.py"),
            cmd("rel_generator", f"{ITEM}/kcp_er", PY, "REL_generator.py"),
        ),
        evaluate=(
            cmd("accuracy_metrics", f"{ITEM}/kcp_er", PY, "evaluate4acc_etc.py"),
            cmd("ranking_metrics", f"{ITEM}/kcp_er", PY, "evaluate4ndcg_etc.py"),
            cmd("top20", f"{ITEM}/kcp_er/datasets", PY, "get_rec_20.py"),
            cmd("proximity", f"{ITEM}/kcp_er", PY, "evaluate_proximity.py"),
            cmd("ep_all", ITEM, PY, "test_ep_all.py"),
            cmd("ep_portion", ITEM, PY, "test_ep_por.py"),
        ),
    ),
    ModelSpec(
        "MulOER-SAN",
        "item",
        aliases=("muloer", "muloer-san"),
        prepare=(
            cmd("expand_predictions", f"{ITEM}/MulOER-SAN/kc_cover", PY, "dataset_process.py"),
            cmd("coverage_transform", f"{ITEM}/MulOER-SAN/kc_cover", PY, "transformer.py"),
            cmd("prediction_transform", f"{ITEM}/MulOER-SAN/select", PY, "chuli.py"),
        ),
        train=(cmd("recommend", f"{ITEM}/MulOER-SAN/select", PY, "rec.py"),),
        evaluate=(
            cmd("build_q_matrix", f"{ITEM}/MulOER-SAN/select", PY, "generate_q_matrix.py"),
            cmd("filter_test", f"{ITEM}/MulOER-SAN/select", PY, "filter_over4_test.py"),
            cmd("map_original_questions", f"{ITEM}/MulOER-SAN/select", PY, "get_original_ex_ques.py"),
            cmd("accuracy_metrics", f"{ITEM}/MulOER-SAN/select", PY, "evaluate_acc.py"),
            cmd("ranking_metrics", f"{ITEM}/MulOER-SAN/select", PY, "evaluate_ndcg.py"),
            cmd("proximity", f"{ITEM}/MulOER-SAN/select", PY, "evaluate_proximity.py"),
            cmd("ep_all", ITEM, PY, "test_ep_all.py"),
            cmd("ep_portion", ITEM, PY, "test_ep_por.py"),
        ),
        notes=("pyKT AKT prediction must already exist under MulOER-SAN/PYKT/data/<dataset>.",),
    ),
    ModelSpec(
        "KG4EX",
        "item",
        aliases=("kg4ex",),
        prepare=(
            cmd("kg_preprocess", f"{ITEM}/KG4EX/get_KG_graph", PY, "preprocess.py"),
            cmd("kg_process", f"{ITEM}/KG4EX/get_KG_graph", PY, "process.py"),
        ),
        train=(
            cmd(
                "train_transe",
                f"{ITEM}/KG4EX/codes",
                PY,
                "run.py",
                "--do_train",
                "{cuda_flag}",
                "--data_path",
                "../data/{dataset}",
                "--model",
                "TransE",
                "-b",
                "{batch_size}",
                "-d",
                "{hidden_dim}",
                "-g",
                "12.0",
                "-a",
                "1.0",
                "-lr",
                "{learning_rate}",
                "-adv",
                "-save",
                "models/{dataset}/TransE_adv",
                "--seed",
                "{seed}",
            ),
            cmd("recommend", f"{ITEM}/KG4EX/codes", PY, "test_TransE.py"),
        ),
        evaluate=(
            cmd("top20", f"{ITEM}/KG4EX/codes/Ep", PY, "ger_rec_20.py"),
            cmd("unique_last", f"{ITEM}/KG4EX/codes/Ep", PY, "get_unique_last.py"),
            cmd("accuracy_ranking_metrics", f"{ITEM}/KG4EX/codes", PY, "evaluate.py"),
            cmd("proximity", f"{ITEM}/KG4EX/codes", PY, "evaluate_proximity.py"),
            cmd("ep_all", ITEM, PY, "test_ep_all.py"),
            cmd("ep_portion", ITEM, PY, "test_ep_por.py"),
        ),
    ),
    ModelSpec(
        "ER-TGA",
        "item",
        aliases=("ertga", "er-tga"),
        prepare=(cmd("preprocess", f"{ITEM}/ER-TGA/code", PY, "Preprocess.py"),),
        train=(
            cmd(
                "train_ertga",
                f"{ITEM}/ER-TGA/code",
                PY,
                "main.py",
                "{ertga_dataset}",
                "{ertga_generations}",
                "{ertga_population}",
            ),
        ),
        evaluate=(
            cmd("build_q_matrix", f"{ITEM}/ER-TGA/code/Ep", PY, "Q.py"),
            cmd("accuracy_metrics", f"{ITEM}/ER-TGA/evaluate", PY, "evaluate4acc_etc.py"),
            cmd("ranking_metrics", f"{ITEM}/ER-TGA/evaluate", PY, "evaluate4ndcg_etc.py"),
            cmd("proximity", f"{ITEM}/ER-TGA/evaluate", PY, "evaluate_proximity.py"),
            cmd("ep_all", ITEM, PY, "test_ep_all.py"),
            cmd("ep_portion", ITEM, PY, "test_ep_por.py"),
        ),
    ),
    ModelSpec(
        "NR4DER",
        "item",
        aliases=("nr4der",),
        prepare=(cmd("build_q_matrix", f"{ITEM}/NR4DER/module", PY, "Q.py"),),
        train=(
            cmd("stage1", f"{ITEM}/NR4DER/main", PY, "stage1_main.py", "--dataset", "{dataset}", "--seed", "{seed}", "--epochs", "{epochs}"),
            cmd("eb_filter", f"{ITEM}/NR4DER", PY, "module/EB_filter.py"),
            cmd("stage2", f"{ITEM}/NR4DER", PY, "main/stage2_main.py", "--dataset", "{dataset}", "--seed", "{seed}", "--epochs", "{epochs}", "--device", "{device}"),
        ),
        evaluate=(
            cmd("accuracy_metrics", f"{ITEM}/NR4DER/module", PY, "evaluate4acc_etc.py"),
            cmd("ranking_metrics", f"{ITEM}/NR4DER/module", PY, "evaluate4ndcg_etc.py"),
            cmd("top20", f"{ITEM}/NR4DER/module", PY, "rec_top20.py"),
            cmd("proximity", f"{ITEM}/NR4DER/module", PY, "evaluate_proximity.py"),
            cmd("ep_all", ITEM, PY, "test_ep_all.py"),
            cmd("ep_portion", ITEM, PY, "test_ep_por.py"),
        ),
    ),
    ModelSpec(
        "SRC",
        "path",
        aliases=("src",),
        prepare=(
            cmd("data_process", f"{PATH}/SRC", PY, "-m", "Unier_cli.path_preprocess", "src"),
            cmd("npz_process", f"{PATH}/SRC", PY, "data/dataprocess_npz.py"),
            cmd("train_dkt", f"{PATH}/SRC", PY, "trainDKT.py", "-d", "{path_dataset}", "-m", "DKT", "--rand_seed", "{seed}"),
            cmd("env_dkt", f"{PATH}/SRC", PY, "data/envDKT.py"),
        ),
        train=(
            cmd(
                "train_src",
                f"{PATH}/SRC",
                PY,
                "trainSRC.py",
                "-d",
                "{path_dataset}",
                "-p",
                "3",
                "--steps",
                "{max_steps}",
                "--target_type",
                "{target_type}",
                "-c",
                "{cuda_index}",
                "--rand_seed",
                "{seed}",
            ),
        ),
        notes=("trainSRC.py performs its existing test pass after training.",),
    ),
    ModelSpec("AC", "path", aliases=("ac",), prepare=path_prepare("AC"), train=path_runsim("AC")),
    ModelSpec("DQN", "path", aliases=("dqn",), prepare=path_prepare("DQN"), train=path_runsim("DQN")),
    ModelSpec(
        "RLTutor",
        "path",
        aliases=("rltutor",),
        prepare=path_prepare("RLTutor", extra=("EduSim/agents/TutoInnerModel.py",)),
        train=path_runsim("RLTutor"),
    ),
    ModelSpec("CSEAL", "path", aliases=("cseal",), prepare=path_prepare("CSEAL"), train=path_runsim("CSEAL")),
    ModelSpec(
        "GEHRL",
        "path",
        aliases=("gehrl",),
        prepare=path_prepare("GEHRL", graph=True),
        train=(
            cmd(
                "train_path_model",
                f"{PATH}/GEHRL/scripts",
                PY,
                "runSim.py",
                "-s",
                "{simulator}",
                "-m",
                "{max_steps}",
                "--max_episode_num",
                "{episodes}",
                "--cuda_device",
                "{cuda_index}",
                "--target_type",
                "{target_type}",
                "--seed",
                "{seed}",
            ),
        ),
    ),
    ModelSpec(
        "DLPR",
        "path",
        aliases=("dlpr",),
        prepare=path_prepare("DLPR", graph=True, extra=("data/dataProcess/prepare_dlpr_envdata.py",)),
        train=(cmd("train_dlpr", f"{PATH}/DLPR/IDALPR", PY, "DLPR.py"),),
        notes=("The external pyKT DIMKT checkpoint remains a required input and is validated before execution.",),
    ),
    ModelSpec("PKSD", "path", aliases=("pksd",), prepare=path_prepare("PKSD", graph=True), train=path_runsim("PKSD")),
    ModelSpec(
        "KnowLP",
        "path",
        aliases=("knowlp",),
        prepare=path_prepare("KnowLP", graph=True, extra=("data/dataProcess/prepare_knowlp_envdata.py",)),
        train=(cmd("train_knowlp", f"{PATH}/KnowLP/IDALPR", PY, "KnowLP.py"),),
        notes=(
            "Generate the KnowLP concept text with EDU-graphRAG/generate_concepts_textgrad.py before running GraphRAG.",
            "GraphRAG outputs and the external pyKT DIMKT checkpoint must be supplied before a full run.",
        ),
    ),
)


_LOOKUP = {}
for _spec in MODEL_SPECS:
    for _key in (_spec.name, *_spec.aliases):
        _LOOKUP[_key.lower().replace("_", "-")] = _spec


def get_model_spec(name: str) -> ModelSpec:
    key = name.lower().replace("_", "-")
    try:
        return _LOOKUP[key]
    except KeyError as exc:
        supported = ", ".join(model_names())
        raise ValueError(f"Unknown model {name!r}. Supported models: {supported}") from exc


def model_names() -> Tuple[str, ...]:
    return tuple(spec.name for spec in MODEL_SPECS)
