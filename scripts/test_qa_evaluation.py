"""CPU-only tests; no torch, downloads, LLM calls or generated code execution."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import qa_eval_utils as utils
import run_qa_multiworker as launcher
import eval_polar_qa as worker


class Tests(unittest.TestCase):
    def test_metrics(self):
        self.assertEqual(utils.score_answer("The Paris.", ["Paris"])["exact_match"], 1)
        self.assertEqual(utils.score_answer("Shakespeare", ["William Shakespeare", "Shakespeare"])["f1"], 1)
        self.assertAlmostEqual(utils.score_answer("William", ["William Shakespeare"])["f1"], 2 / 3)
        self.assertEqual(utils.score_answer("London", ["Paris"])["f1"], 0)
        self.assertEqual(utils.score_answer("北京", ["北京"])["f1"], 1)
        self.assertEqual(utils.tokens("Beijing北京"), ["beijing", "北", "京"])
        self.assertEqual(utils.score_answer("", ["Paris"])["exact_match"], 0)

    def test_choice_extraction(self):
        choices = {"A": "a", "B": "b"}
        self.assertEqual(utils.extract_answer("Answer: B", "multiple_choice", choices), "B")
        self.assertEqual(utils.extract_answer("A\nB", "multiple_choice", choices), "")
        self.assertEqual(utils.extract_answer("I think B is correct", "multiple_choice", choices), "")
        self.assertEqual(utils.extract_answer("C", "multiple_choice", choices), "")
        self.assertEqual(utils.score_answer("B", ["B"], "multiple_choice")["exact_match"], 1)

    def test_schema(self):
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "tasks.json"
            utils.write_json(source, [
                {"id": "one", "question": "Where?", "answers": {"text": ["London"]}},
                {"id": "two", "query": "Who?", "answer": {"aliases": ["Ada"], "value": "Ada Lovelace"}},
            ])
            rows = utils.read_tasks(source)
            self.assertEqual(rows[0]["answers"], ["London"])
            self.assertEqual(rows[1]["answers"], ["Ada", "Ada Lovelace"])
            utils.write_json(source, [
                {"question": "Which?", "choices": {"label": ["A", "B"], "text": ["one", "two"]},
                 "answerKey": "B"}])
            self.assertEqual(utils.read_tasks(source, "multiple_choice")[0]["answers"], ["B"])
            utils.write_json(source, [{"question": "Q", "answers": []}])
            with self.assertRaises(ValueError):
                utils.read_tasks(source)
            utils.write_json(source, [{"id": "1", "question": "Q", "answer": "X"}] * 2)
            with self.assertRaises(ValueError):
                utils.read_tasks(source)

    def test_prompt_and_paths(self):
        task = {"question": "Who?", "context": "People", "answers": ["SECRET_REFERENCE"], "choices": {}}
        self.assertNotIn("SECRET_REFERENCE", utils.model_prompt(task))
        desc = utils.describe_path([0, 1, 0, 1, 3], 5)
        self.assertEqual(desc["skipped_layers"], [2, 4])
        self.assertEqual(desc["repeated_layers"], [0, 1])
        self.assertEqual(desc["operations"][2]["operation"], "repeat")

    @staticmethod
    def result(task_id):
        return {
            "task_id": task_id, "dataset_index": 0,
            "base": {"exact_match": 0., "f1": .5},
            "candidates": [
                {"exact_match": 0., "f1": .5, "unique_depth": 30, "total_layer_executions": 40},
                {"exact_match": 1., "f1": 1., "unique_depth": 32, "total_layer_executions": 36},
            ],
        }

    def test_oracle(self):
        summary = utils.summarize([self.result("q1")])
        self.assertEqual(summary["base"]["exact_match"], 0)
        self.assertEqual(summary["polar_top1"]["exact_match"], 0)
        self.assertEqual(summary["polar_topk_oracle"]["exact_match"], 1)
        row = self.result("q2")
        row["candidates"] = []
        self.assertEqual(utils.summarize([row])["polar_top1"]["exact_match"], 0)

    def test_completed_worker_resume_without_gpu(self):
        with tempfile.TemporaryDirectory() as td:
            source, output = Path(td) / "input.json", Path(td) / "result.json"
            utils.write_json(source, [{"id": "q1", "question": "Q", "answer": "Paris"}])
            argv = ["worker", "--model_path", "/models/qwen", "--checkpoint_path", "/predictor.pt",
                    "--input_json", str(source), "--save_json", str(output), "--resume_existing"]
            args = worker.build_parser().parse_args(argv[1:])
            config = {k: v for k, v in vars(args).items() if k != "resume_existing"}
            config["tasks_sha256"] = hashlib.sha256(json.dumps(
                utils.read_tasks(source), sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            utils.write_json(output, {"configuration": config, "results": [self.result("q1")]})
            with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()):
                worker.main()
            self.assertEqual(json.loads(output.read_text())["status"], "complete")
            with patch("sys.argv", argv + ["--max_new_tokens", "2048"]):
                with self.assertRaises(ValueError):
                    worker.main()

    def test_launcher_sharding_resume_and_missing(self):
        with tempfile.TemporaryDirectory() as td:
            source, out = Path(td) / "tasks.json", Path(td) / "out"
            rows = [{"id": f"qa/{i}", "question": f"Question {i}", "answer": "Paris"} for i in range(17)]
            utils.write_json(source, rows)
            calls = []
            outer = self

            class FakeProcess:
                def __init__(self, cmd, **kwargs):
                    calls.append((cmd, kwargs))
                    self.pid = len(calls)
                    self.returncode = 0
                    args = worker.build_parser().parse_args(cmd[3:])
                    tasks = utils.read_tasks(args.input_json)[args.start_idx:args.start_idx + args.num_samples]
                    utils.write_json(args.save_json, {"status": "complete",
                                                      "results": [outer.result(t["task_id"]) for t in tasks]})

                def wait(self):
                    return self.returncode

                def poll(self):
                    return self.returncode

            argv = ["launcher", "--model_path", "/models/qwen",
                    "--checkpoint_path", "/predictor.pt", "--input_json", str(source),
                    "--output_dir", str(out), "--gpus", "0 1 2 3", "--workers_per_gpu", "3",
                    "--max_total_workers", "8", "--max_new_tokens", "2048"]
            with patch("sys.argv", argv), patch.object(launcher.subprocess, "Popen", FakeProcess):
                with contextlib.redirect_stdout(io.StringIO()):
                    launcher.main()
            manifest = json.loads((out / "manifest.json").read_text())
            counts = [s["num_samples"] for s in manifest["shards"]]
            self.assertEqual(sum(counts), 17)
            self.assertLessEqual(max(counts) - min(counts), 1)
            self.assertEqual([s["gpu"] for s in manifest["shards"]], ["0", "1", "2", "3"] * 2)
            self.assertEqual(json.loads((out / "summary.json").read_text())["completed_tasks"], 17)
            for cmd, kwargs in calls:
                self.assertIn("2048", cmd)
                self.assertIn(kwargs["env"]["CUDA_VISIBLE_DEVICES"], ["0", "1", "2", "3"])
            (out / manifest["shards"][0]["result_file"]).unlink()
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertFalse(launcher.merge_results(out, manifest))
            summary = json.loads((out / "summary.json").read_text())
            self.assertEqual(summary["completed_tasks"], 15)
            self.assertEqual(len(summary["missing_task_ids"]), 2)
            with patch("sys.argv", argv + ["--resume"]), patch.object(launcher.subprocess, "Popen", FakeProcess):
                with contextlib.redirect_stdout(io.StringIO()):
                    launcher.main()
            self.assertTrue(all("--resume_existing" in cmd for cmd, _ in calls[8:]))
            with patch("sys.argv", argv + ["--resume", "--limit", "8"]):
                with self.assertRaises(ValueError), contextlib.redirect_stdout(io.StringIO()):
                    launcher.main()


if __name__ == "__main__":
    unittest.main()
