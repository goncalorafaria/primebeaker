"""Use vLLM's multiprocess-aware CLI while retaining LiteRegistry lifecycle."""
import sys
import fire
from literegistry.services import vllm_wrapper


def server_command(self):
    if self.runtime.name == 'local':
        return [sys.executable, '-m', 'vllm.entrypoints.cli.main', 'serve']
    return ['vllm', 'serve']


def model_args(self):
    return [self.model]


if __name__ == '__main__':
    vllm_wrapper.VLLMServerManager.get_server_command = server_command
    vllm_wrapper.VLLMServerManager.get_model_args = model_args
    fire.Fire(vllm_wrapper.main)
