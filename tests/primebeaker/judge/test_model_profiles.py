import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from primebeaker.judge.model_profiles import load_model_profile


def _config(directory: Path) -> SimpleNamespace:
    return SimpleNamespace(
        model_profiles_dir=str(directory),
        prompt_template_path="unused.json",
        max_tool_calls=8,
        max_tokens=1024,
        timeout=90.0,
        max_retries=3,
        rubric_max_retries=1,
        rollout_timeout=3600.0,
        tools=("terminal",),
        tool_server_url="http://localhost:8080",
    )


class ModelProfileTest(unittest.TestCase):
    def test_exact_model_profile_selects_template_and_turns(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            (directory / "template.json").write_text("{}", encoding="utf-8")
            (directory / "judge.json").write_text(
                json.dumps(
                    {
                        "model": "org/judge-model",
                        "rubric_mode": "grouped_rubrics",
                        "prompt_template_path": "template.json",
                        "max_turns": 4,
                        "max_tool_calls": 3,
                        "max_tokens": 512,
                        "rubric_max_retries": 2,
                        "evaluation_timeout": 500,
                        "tools": ["terminal"],
                    }
                ),
                encoding="utf-8",
            )

            profile = load_model_profile("org/judge-model", _config(directory))
        self.assertEqual(profile.rubric_mode, "grouped_rubrics")

        self.assertEqual(profile.max_turns, 4)
        self.assertEqual(profile.max_tool_calls, 3)
        self.assertEqual(profile.max_tokens, 512)
        self.assertEqual(profile.rubric_max_retries, 2)
        self.assertEqual(profile.evaluation_timeout, 500)
        self.assertEqual(profile.tools, ("terminal",))

    def test_invalid_rubric_mode_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            (directory / "template.json").write_text("{}", encoding="utf-8")
            (directory / "judge.json").write_text(
                json.dumps(
                    {
                        "model": "org/judge-model",
                        "rubric_mode": "all_at_once-ish",
                        "prompt_template_path": "template.json",
                        "max_turns": 1,
                        "max_tool_calls": 0,
                        "tools": [],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "rubric_mode"):
                load_model_profile("org/judge-model", _config(directory))


    def test_checked_in_profile_inherits_the_worker_gateway_url(self):
        from primebeaker.judge_catalog import judge_profile_catalog_dir
        directory = judge_profile_catalog_dir()
        config = _config(directory)
        config.tool_server_url = "http://127.0.0.1:57121"

        profile = load_model_profile(
            "/weka/gfaria/prime_sft/outputs/"
            "qwen35_4b_glm52_sft_c0e8854bb5143cac_rl_5k_161bc7548bdec992_"
            "step400-nooversamp-renderer-eval-inflight512-kl1e-3-lr1e-6/"
            "weights/step_400",
            config,
        )

        self.assertEqual(profile.tool_server_url, "http://127.0.0.1:57121")

    def test_missing_model_profile_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory_name:
            with self.assertRaisesRegex(ValueError, "no model profile found"):
                load_model_profile("missing", _config(Path(directory_name)))


if __name__ == "__main__":
    unittest.main()
