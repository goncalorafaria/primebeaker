"""Use PrimeRL's exporter with this run's resolved-config location."""
import importlib.util
import os
from pathlib import Path
spec = importlib.util.spec_from_file_location('dcp_export', '/weka/gfaria/prime-rl-mc32/tools/convert_dcp_to_bf16.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.get_config_dir = lambda _: Path(os.environ.get('RUN_CONFIG_DIR', '/weka/gfaria/search-agent-repr/outputs/search-agent-mc32-16train-16infer-cp4-resume130/configs'))
original_setup_model = module.setup_model
def setup_model(*args, **kwargs):
    model = original_setup_model(*args, **kwargs)
    # Qwen3.5 may declare this optional Transformers metadata as None.
    if getattr(model, '_tied_weights_keys', None) is None:
        model._tied_weights_keys = []
    # PrimeRL's FSDP wrapper is not a Transformers GenerationMixin.
    # Export generation defaults from the model config when metadata is absent.
    if getattr(model, 'generation_config', None) is None:
        from transformers import GenerationConfig
        model.generation_config = GenerationConfig.from_model_config(model.config)
    return model
module.setup_model = setup_model
module.main()
