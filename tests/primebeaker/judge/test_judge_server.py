import asyncio
import tempfile
import threading
import unittest
from unittest.mock import patch

from primebeaker.judge.judge_server import JudgeRequest, JudgeServer, JudgeServerConfig, _config_from_environment, main


class JudgeServerTest(unittest.TestCase):
    def test_request_requires_the_four_public_fields(self):
        with self.assertRaisesRegex(ValueError, "missing required field"):
            JudgeRequest.from_json({"input": "question"})

    def test_request_rejects_empty_rubric_text(self):
        with self.assertRaisesRegex(ValueError, "each rubric"):
            JudgeRequest.from_json(
                {"input": "question", "output": "answer", "rubrics": [""], "model": "model"}
            )

    def test_requests_do_not_wait_for_a_shared_worker_slot(self):
        async def check():
            count = 40
            entered = set()
            release = threading.Event()
            all_entered = asyncio.Event()
            loop = asyncio.get_running_loop()

            def mark(index):
                entered.add(index)
                if len(entered) == count:
                    all_entered.set()

            async def workflow(request, *, executor, **kwargs):
                def work():
                    loop.call_soon_threadsafe(mark, request.input)
                    release.wait(10)
                    return {"judgments": []}
                return await loop.run_in_executor(executor, work)

            class Request:
                def __init__(self, index):
                    self.index = index

                async def json(self):
                    return {"input": str(self.index), "output": "answer", "rubrics": ["criterion"], "model": "model"}

            with tempfile.TemporaryDirectory() as directory:
                server = JudgeServer(JudgeServerConfig(registry=directory))
                with patch("primebeaker.judge.judge_server.run_verifier_tool_judge", side_effect=workflow):
                    tasks = [asyncio.create_task(server.judge(Request(i))) for i in range(count)]
                    try:
                        await asyncio.wait_for(all_entered.wait(), timeout=5)
                    finally:
                        release.set()
                        results = await asyncio.gather(*tasks)
                        await server.store.close()
                    self.assertEqual(len(entered), count)
                    self.assertTrue(all(result.status_code == 200 for result in results))

        asyncio.run(check())

    def test_multi_worker_uses_uvicorn_factory(self):
        with patch("primebeaker.judge.judge_server.uvicorn.run") as run, patch.dict("os.environ", clear=False):
            main(
                port=9191,
                registry="redis://registry",
                workers=3,
                rubric_max_retries=2,
            )
            self.assertEqual(_config_from_environment().registry, "redis://registry")
            self.assertEqual(_config_from_environment().rubric_max_retries, 2)

        run.assert_called_once_with(
            "primebeaker.judge.judge_server:create_app",
            host="0.0.0.0",
            port=9191,
            workers=3,
            factory=True,
            log_level="info",
            access_log=False,
        )


if __name__ == "__main__":
    unittest.main()


def test_worker_configuration_preserves_local_gateway_in_child_processes():
    from primebeaker.judge.judge_server import _publish_worker_configuration
    with patch.dict('os.environ', clear=True):
        config = JudgeServerConfig(model_gateway_url='http://127.0.0.1:57121')
        _publish_worker_configuration(config)
        assert _config_from_environment() == config


def test_default_resources_are_installed_and_profiles_load():
    from pathlib import Path
    from primebeaker.judge.model_profiles import load_model_profile
    from primebeaker.judge_catalog import list_judge_model_profiles
    config = JudgeServerConfig()
    assert Path(config.prompt_template_path).is_file()
    profiles = list_judge_model_profiles(config.model_profiles_dir)
    assert profiles
    for profile in profiles:
        assert Path(load_model_profile(profile.model, config).prompt_template_path).is_file()
