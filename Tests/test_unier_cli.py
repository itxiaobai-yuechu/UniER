from __future__ import annotations

import argparse
import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from Unier_cli.data import DatasetMaterializer, canonical_layout, data_assets_for
from Unier_cli.metrics import parse_metrics
from Unier_cli.path_preprocess import prepare_src
from Unier_cli.postprocess import dre_postprocess
from Unier_cli.registry import MODEL_SPECS, get_model_spec
from Unier_cli.runner import normalize_dataset, render_command, render_values, stage_commands


def args(**overrides):
    defaults = dict(
        dataset="assist2017",
        python="python",
        device="cpu",
        seed=42,
        epochs=3,
        episodes=2,
        max_steps=10,
        batch_size=None,
        learning_rate=None,
        hidden_dim=1000,
        concept_count=None,
        target_type="all",
        top_k="1,3,5,10,20",
        ertga_generations=10,
        ertga_population=12,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class RegistryTests(unittest.TestCase):
    def test_all_eighteen_models_registered(self):
        self.assertEqual(18, len(MODEL_SPECS))
        self.assertEqual("KCP-ER", get_model_spec("kcp_er").name)
        self.assertEqual("SimpleKT", get_model_spec("simplekt").name)

    def test_dataset_aliases(self):
        self.assertEqual("assist2017", normalize_dataset("assist17"))
        self.assertEqual("assist2009", normalize_dataset("assist2009"))

    def test_every_registered_python_script_exists(self):
        repo = Path(__file__).resolve().parents[1]
        for spec in MODEL_SPECS:
            values = render_values(args(), spec)
            for stage in (spec.prepare, spec.train, spec.evaluate):
                for command in stage:
                    argv = render_command(command, values)
                    if len(argv) >= 2 and argv[1].endswith(".py"):
                        script = repo / command.cwd / argv[1]
                        self.assertTrue(script.exists(), f"{spec.name}: {script}")

    def test_dependency_sensitive_metric_order(self):
        dre = [command.label for command in get_model_spec("DRE").evaluate]
        kcper = [command.label for command in get_model_spec("KCP-ER").evaluate]
        muloer = [command.label for command in get_model_spec("MulOER-SAN").evaluate]
        nr4der = [command.label for command in get_model_spec("NR4DER").evaluate]
        self.assertLess(dre.index("dre_postprocess"), dre.index("proximity"))
        self.assertLess(kcper.index("top20"), kcper.index("proximity"))
        self.assertLess(muloer.index("filter_test"), muloer.index("proximity"))
        self.assertLess(muloer.index("map_original_questions"), muloer.index("proximity"))
        self.assertLess(nr4der.index("top20"), nr4der.index("proximity"))

    def test_dkt_sort_replaces_only_item_ep_commands(self):
        for spec in (model for model in MODEL_SPECS if model.level == "item"):
            item_commands = stage_commands(spec, "evaluate", dkt_sort=True)
            scripts = {command.label: command.argv[1] for command in item_commands}
            self.assertEqual("test_ep_all_update.py", scripts["ep_all"], spec.name)
            self.assertEqual("test_ep_por_update.py", scripts["ep_portion"], spec.name)

        kcper_scripts = {
            command.label: command.argv[1]
            for command in stage_commands(get_model_spec("KCP-ER"), "evaluate", dkt_sort=True)
        }
        self.assertEqual("evaluate_proximity.py", kcper_scripts["proximity"])

        original = stage_commands(get_model_spec("KCP-ER"), "evaluate", dkt_sort=False)
        original_scripts = {command.label: command.argv[1] for command in original}
        self.assertEqual("test_ep_all.py", original_scripts["ep_all"])
        self.assertEqual("test_ep_por.py", original_scripts["ep_portion"])

        path_commands = stage_commands(get_model_spec("AC"), "train", dkt_sort=True)
        self.assertEqual(get_model_spec("AC").train, path_commands)

    def test_all_models_declare_canonical_inputs(self):
        for spec in MODEL_SPECS:
            self.assertTrue(data_assets_for(spec), spec.name)
        layout = canonical_layout(MODEL_SPECS)
        self.assertIn("train_valid.csv", layout)
        self.assertIn("models/akt/qid_test_predictions.txt", layout)
        self.assertIn("dataRec", layout)
        self.assertIn("models/ac/agent_weights/ValBest.ckpt", layout)


class UnifiedDataTests(unittest.TestCase):
    def test_common_inputs_are_hardlinked_from_the_unified_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp)
            dataset_dir = repo / "datasets" / "assist2017"
            dataset_dir.mkdir(parents=True)
            (dataset_dir / "train_valid.csv").write_text("uid,questions\n", encoding="utf-8")
            (dataset_dir / "test.csv").write_text("uid,questions\n", encoding="utf-8")

            spec = get_model_spec("DRE")
            values = render_values(args(), spec)
            manager = DatasetMaterializer(repo, repo / "datasets", values, spec)
            manager.materialize("prepare")

            target = repo / "Item-level Exercise Recommendation" / "DRE" / "data" / "assist2017"
            self.assertTrue(os.path.samefile(dataset_dir / "train_valid.csv", target / "train_valid.csv"))
            self.assertTrue(os.path.samefile(dataset_dir / "test.csv", target / "test.csv"))

    def test_mutable_model_input_is_copied(self):
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp)
            dataset_dir = repo / "datasets" / "assist2017"
            dataset_dir.mkdir(parents=True)
            for name in ("test_sequences.csv", "pkm.pth", "pkc.pth"):
                (dataset_dir / name).write_text(name, encoding="utf-8")

            spec = get_model_spec("KCP-ER")
            values = render_values(args(), spec)
            manager = DatasetMaterializer(repo, repo / "datasets", values, spec)
            manager.materialize("prepare")

            target = (
                repo
                / "Item-level Exercise Recommendation"
                / "kcp_er"
                / "datasets"
                / "assist2017"
                / "select"
                / "test_sequences.csv"
            )
            self.assertEqual("test_sequences.csv", target.read_text(encoding="utf-8"))
            self.assertFalse(os.path.samefile(dataset_dir / "test_sequences.csv", target))

    def test_missing_input_error_points_to_only_the_unified_root(self):
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp)
            spec = get_model_spec("DRE")
            values = render_values(args(), spec)
            manager = DatasetMaterializer(repo, repo / "datasets", values, spec)
            with self.assertRaises(FileNotFoundError) as caught:
                manager.materialize("prepare")
            message = str(caught.exception)
            self.assertIn(str(repo / "datasets" / "assist2017" / "train_valid.csv"), message)
            self.assertNotIn("Item-level Exercise Recommendation", message)

    def test_src_adapter_builds_original_dataoff_datarec_split(self):
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp)
            data_dir = project / "data" / "assist17"
            data_dir.mkdir(parents=True)
            rows = {
                "concepts": ["1,2", "2,3", "3,4", "4,5"],
                "responses": ["1,0", "0,1", "1,1", "0,0"],
            }
            pd.DataFrame(rows).iloc[:2].to_csv(data_dir / "train_valid.csv", index=False)
            pd.DataFrame(rows).iloc[2:].to_csv(data_dir / "test.csv", index=False)

            prepare_src("assist17", seed=42, project_root=project)

            data_off = (data_dir / "dataOff").read_text(encoding="utf-8").splitlines()
            data_rec = (data_dir / "dataRec").read_text(encoding="utf-8").splitlines()
            self.assertEqual(3, len(data_off))
            self.assertEqual(1, len(data_rec))


class MetricsTests(unittest.TestCase):
    def test_parse_combined_metrics(self):
        output = """
acc mean:0.800 std:0.100
k = 5, ndcg: 0.700, map: 0.600, mrr: 0.500, precision: 0.400, f1: 0.300, recall: 0.200
Recommendation Length =  5 | Valid Students = 10 | Average Ep = 0.1250
Proximity Mean: 0.900
Proximity Std: 0.020
"""
        parsed = parse_metrics(output, "ep_all")
        self.assertEqual(0.8, parsed["acc"])
        self.assertEqual(0.7, parsed["ndcg@5"])
        self.assertEqual(0.125, parsed["ep_all@5"])
        self.assertEqual(0.9, parsed["proximity"])

    def test_parse_mmer_at_k(self):
        parsed = parse_metrics("@3\tNDCG:0.71\tHit:0.62\tF1:0.53\tMAP:0.44\tMRR:0.35", "ranking_metrics")
        self.assertEqual(0.71, parsed["ndcg@3"])
        self.assertEqual(0.44, parsed["map@3"])

    def test_parse_dynamic_dkt_ep(self):
        output = "step= 5 valid_students=    10 mean_Ep=0.125000"
        self.assertEqual(0.125, parse_metrics(output, "ep_all")["ep_all@5"])
        self.assertEqual(0.125, parse_metrics(output, "ep_portion")["ep_portion@5"])

    def test_parse_path_mean_reward_not_learner_count(self):
        output = "Agent test mean rewards on learner nums: [1000] rewards: [-0.06230029]"
        parsed = parse_metrics(output, "train_path_model")
        self.assertEqual(-0.06230029, parsed["mean_reward"])


class DREPostprocessTests(unittest.TestCase):
    def test_postprocess_matches_original_three_step_behavior(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "Item-level Exercise Recommendation" / "DRE" / "data" / "assist2017"
            data.mkdir(parents=True)
            recommendations = [f"{uid}\t{uid},{uid + 1}" for uid in range(8)]
            (data / "recommended_problems.txt").write_text("\n".join(recommendations) + "\n", encoding="utf-8")
            pd.DataFrame({"uid": [0, 1, 2], "questions": ["1", "2", "3"]}).to_csv(
                data / "train_valid.csv", index=False
            )
            pd.DataFrame({"uid": [3, 4], "questions": ["4", "5"]}).to_csv(data / "test.csv", index=False)

            outputs = dre_postprocess(root, "assist2017")

            final_lines = outputs["recommended_problems_final"].read_text(encoding="utf-8").splitlines()
            self.assertEqual(recommendations[:2], final_lines)
            full = pd.read_csv(outputs["full_data"])
            filtered = pd.read_csv(outputs["filtered_full_data"])
            self.assertEqual(5, len(full))
            self.assertEqual([0, 1], filtered["uid"].tolist())


if __name__ == "__main__":
    unittest.main()
