"""CPU-only tests; never load an LLM or execute generated code."""
import ast
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("launcher", HERE / "run_humaneval_multiworker.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class Tests(unittest.TestCase):
    def test_launch_balanced_resume_and_merge(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "tasks.json"
            rows = [{"task_id": f"test/{i}", "prompt": "def f():", "test": "test",
                     "entry_point": "f"} for i in range(17)]
            launcher.write_json(source, rows)
            out = root / "out"
            commands = []

            class FakeProcess:
                def __init__(self, cmd, **kwargs):
                    commands.append((cmd, kwargs))
                    self.pid = len(commands)
                    self.returncode = 0
                    pairs = dict(zip(cmd[3::2], cmd[4::2]))
                    start = int(pairs["--start_idx"])
                    end = start + int(pairs["--num_samples"])
                    launcher.write_json(pairs["--save_json"], {
                        "status": "complete",
                        "results": [{"task_id": r["task_id"],
                                     "base": {"passed": True},
                                     "polar_top1": False, "polar_topk": True,
                                     "candidates": [{"total_layer_executions": 40,
                                                     "unique_depth": 30}]}
                                    for r in rows[start:end]]})

                def wait(self):
                    return self.returncode

                def poll(self):
                    return self.returncode

            argv = ["launcher", "--model_path", "/models/qwen",
                    "--checkpoint_path", "/predictor.pt", "--input_json", str(source),
                    "--output_dir", str(out), "--gpus", "0 1 2 3",
                    "--workers_per_gpu", "3", "--max_total_workers", "8",
                    "--allow_code_execution"]
            with patch("sys.argv", argv), patch.object(launcher.subprocess, "Popen", FakeProcess):
                with contextlib.redirect_stdout(io.StringIO()):
                    launcher.main()
            manifest = json.loads((out / "manifest.json").read_text())
            counts = [x["num_samples"] for x in manifest["shards"]]
            self.assertEqual(sum(counts), 17)
            self.assertLessEqual(max(counts) - min(counts), 1)
            self.assertEqual([x["gpu"] for x in manifest["shards"]], ["0", "1", "2", "3"] * 2)
            summary = json.loads((out / "summary.json").read_text())
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["base_accuracy_on_completed"], 1.0)
            self.assertEqual(summary["polar_top1_accuracy_on_completed"], 0.0)
            # Missing shard must be visible and not silently counted as a full run.
            (out / manifest["shards"][0]["result_file"]).unlink()
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertFalse(launcher.merge_results(out, manifest))
            summary = json.loads((out / "summary.json").read_text())
            self.assertEqual(summary["completed_tasks"], 15)
            self.assertEqual(len(summary["missing_task_ids"]), 2)
            with patch("sys.argv", argv + ["--resume"]), patch.object(launcher.subprocess, "Popen", FakeProcess):
                with contextlib.redirect_stdout(io.StringIO()):
                    launcher.main()
            self.assertTrue(all("--resume_existing" in cmd for cmd, _ in commands[8:]))

    def test_path_description(self):
        # Extract only the pure helper, so no torch/transformers are needed.
        tree = ast.parse((HERE / "eval_polar_humaneval.py").read_text())
        func = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == "describe_path")
        func.returns = None
        for arg in func.args.args:
            arg.annotation = None
        namespace = {}
        exec(compile(ast.Module(body=[func], type_ignores=[]), "<helper>", "exec"), namespace)
        desc = namespace["describe_path"]([0, 1, 0, 1, 3], 5)
        self.assertEqual(desc["repeated_layers"], [0, 1])
        self.assertEqual(desc["skipped_layers"], [2, 4])
        self.assertEqual(desc["operations"][2]["operation"], "repeat")


if __name__ == "__main__":
    unittest.main()
