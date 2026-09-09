"""Lossless, typed editing for Prime-RL SFT and RL TOML documents."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tomllib
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


TrainingKind = Literal["sft", "rl"]


def load_toml(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise TypeError(f"unsupported TOML value type: {type(value).__name__}")


def dump_toml_document(config: dict[str, Any]) -> str:
    """Serialize a complete TOML document, including arrays of tables."""

    lines: list[str] = []

    def emit(table: dict[str, Any], prefix: tuple[str, ...], header: str | None) -> None:
        if header is not None:
            if lines:
                lines.append("")
            lines.append(header)
        arrays = {
            key: value
            for key, value in table.items()
            if isinstance(value, list)
            and value
            and all(isinstance(item, dict) for item in value)
        }
        mappings = {key: value for key, value in table.items() if isinstance(value, dict)}
        for key, value in table.items():
            if key not in mappings and key not in arrays:
                lines.append(f"{key} = {_toml_value(value)}")
        for key, value in mappings.items():
            child = prefix + (key,)
            emit(value, child, "[" + ".".join(child) + "]")
        for key, entries in arrays.items():
            child = prefix + (key,)
            for entry in entries:
                emit(entry, child, "[[" + ".".join(child) + "]]" )

    emit(config, (), None)
    return "\n".join(lines) + "\n"


class TrainingData(BaseModel):
    """Dataset paths needed to bind one training TOML."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    kind: TrainingKind
    train_path: str = Field(min_length=1)
    validation_path: str = Field(min_length=1)
    hf_dataset_path: str | None = None
    environment_args: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _sft_dataset(self) -> "TrainingData":
        if self.kind == "sft" and not self.hf_dataset_path:
            raise ValueError("SFT training data requires hf_dataset_path")
        return self


# Backward-friendly name for callers passing an artifact manifest.
DataArtifacts = TrainingData


class TrainingToml(BaseModel):
    """Immutable wrapper around a complete parsed training TOML."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    template_path: Path
    values: dict[str, Any]

    @classmethod
    def from_path(cls, path: str | Path) -> "TrainingToml":
        template_path = Path(path).resolve()
        if not template_path.is_file():
            raise FileNotFoundError(template_path)
        return cls(template_path=template_path, values=load_toml(template_path))

    def write(self, path: str | Path) -> Path:
        output = Path(path)
        if output.suffix.lower() != ".toml":
            raise ValueError("training TOML output must end in .toml")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(dump_toml_document(self.values), encoding="utf-8")
        return output

    def with_output_dir(self, output_dir: str | Path) -> "TrainingToml":
        value = str(output_dir)
        if not value:
            raise ValueError("output_dir must not be empty")
        values = deepcopy(self.values)
        if "output_dir" not in values:
            raise ValueError("training TOML needs a top-level output_dir")
        values["output_dir"] = value
        return self.model_copy(update={"values": values})

    def with_wandb_tags(self, *tags: str, replace: bool = False) -> "TrainingToml":
        if not tags or any(not tag for tag in tags):
            raise ValueError("at least one non-empty W&B tag is required")
        values = deepcopy(self.values)
        wandb = values.get("wandb")
        if not isinstance(wandb, dict):
            raise ValueError("training TOML needs a [wandb] table")
        current = wandb.get("tags", [])
        if not isinstance(current, list) or not all(isinstance(tag, str) for tag in current):
            raise ValueError("wandb.tags must be an array of strings")
        wandb["tags"] = list(dict.fromkeys(tags if replace else [*current, *tags]))
        return self.model_copy(update={"values": values})

    def with_wandb_name(self, name: str) -> "TrainingToml":
        if not name:
            raise ValueError("W&B run name must not be empty")
        values = deepcopy(self.values)
        wandb = values.get("wandb")
        if not isinstance(wandb, dict):
            raise ValueError("training TOML needs a [wandb] table")
        wandb["name"] = name
        return self.model_copy(update={"values": values})


class SFTTrainingToml(TrainingToml):
    @classmethod
    def from_path(cls, path: str | Path) -> "SFTTrainingToml":
        parsed = super().from_path(path)
        data = parsed.values.get("data")
        validation = parsed.values.get("val")
        validation_data = validation.get("data") if isinstance(validation, dict) else None
        if not isinstance(data, dict) or not isinstance(validation_data, dict):
            raise ValueError("SFT TOML needs [data] and [val.data] tables")
        if data.get("type") != "sft":
            raise ValueError("SFT TOML needs data.type = 'sft'")
        return cls(template_path=parsed.template_path, values=parsed.values)

    def bind_data(self, data: TrainingData) -> "SFTTrainingToml":
        if data.kind != "sft" or not data.hf_dataset_path:
            raise ValueError("SFT TOML requires SFT data with hf_dataset_path")
        values = deepcopy(self.values)
        values["data"].update(
            {"type": "sft", "name": data.hf_dataset_path, "splits": ["train"]}
        )
        values["val"]["data"].update(
            {"type": "sft", "name": data.hf_dataset_path, "splits": ["validation"]}
        )
        return self.model_copy(update={"values": values})


class RLTrainingToml(TrainingToml):
    @classmethod
    def from_path(cls, path: str | Path) -> "RLTrainingToml":
        parsed = super().from_path(path)
        if not isinstance(parsed.values.get("orchestrator"), dict):
            raise ValueError("RL TOML needs an [orchestrator] table")
        return cls(template_path=parsed.template_path, values=parsed.values)

    def with_model_checkpoint(self, checkpoint: str | Path) -> "RLTrainingToml":
        value = str(checkpoint)
        if not value:
            raise ValueError("checkpoint must not be empty")
        values = deepcopy(self.values)
        for table_path in (
            ("trainer", "model"),
            ("orchestrator", "model"),
            ("inference", "model"),
        ):
            table: object = values
            for key in table_path:
                table = table.get(key) if isinstance(table, dict) else None
            if not isinstance(table, dict):
                raise ValueError(f"RL TOML needs [{'.'.join(table_path)}] table")
            table["name"] = value
        return self.model_copy(update={"values": values})

    def with_learning_rate(self, learning_rate: float) -> "RLTrainingToml":
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        values = deepcopy(self.values)
        optim = values.get("trainer", {}).get("optim")
        if not isinstance(optim, dict):
            raise ValueError("RL TOML needs a [trainer.optim] table")
        optim["lr"] = learning_rate
        return self.model_copy(update={"values": values})

    def with_kl_tau(self, kl_tau: float) -> "RLTrainingToml":
        if kl_tau < 0:
            raise ValueError("kl_tau must be non-negative")
        values = deepcopy(self.values)
        loss = values.get("trainer", {}).get("loss")
        if not isinstance(loss, dict):
            raise ValueError("RL TOML needs a [trainer.loss] table")
        loss["kl_tau"] = kl_tau
        return self.model_copy(update={"values": values})

    def with_oversampling_factor(self, factor: int) -> "RLTrainingToml":
        if factor < 1:
            raise ValueError("oversampling_factor must be at least one")
        values = deepcopy(self.values)
        orchestrator = values.get("orchestrator")
        if not isinstance(orchestrator, dict):
            raise ValueError("RL TOML needs an [orchestrator] table")
        orchestrator["oversampling_factor"] = factor
        return self.model_copy(update={"values": values})

    def with_eval_num_examples(self, num_examples: int) -> "RLTrainingToml":
        if num_examples <= 0:
            raise ValueError("num_examples must be positive")
        values = deepcopy(self.values)
        sources = values.get("orchestrator", {}).get("eval", {}).get("source")
        if not isinstance(sources, list) or not sources:
            raise ValueError("RL TOML needs [[orchestrator.eval.source]] tables")
        for source in sources:
            if not isinstance(source, dict):
                raise ValueError("RL source entries must be TOML tables")
            source["num_examples"] = num_examples
        return self.model_copy(update={"values": values})

    def with_environment_namespace(
        self,
        namespace: str = "primebeaker.environments",
    ) -> "RLTrainingToml":
        """Point every Prime-RL legacy environment at this installed package."""

        from primebeaker.environments.registry import ENVIRONMENTS

        bundled_modules = set(ENVIRONMENTS.values())

        if not namespace:
            raise ValueError("environment namespace must not be empty")
        values = deepcopy(self.values)
        orchestrator = values.get("orchestrator")
        if not isinstance(orchestrator, dict):
            raise ValueError("RL TOML needs an [orchestrator] table")
        changed = 0
        for split in ("train", "eval"):
            section = orchestrator.get(split)
            if not isinstance(section, dict):
                continue
            for collection in ("source", "env"):
                entries = section.get(collection)
                if not isinstance(entries, list):
                    continue
                for entry in entries:
                    legacy = entry.get("legacy") if isinstance(entry, dict) else None
                    environment_id = legacy.get("id") if isinstance(legacy, dict) else None
                    if not isinstance(environment_id, str) or not environment_id:
                        continue
                    module = environment_id.rsplit(".", 1)[-1]
                    if module not in bundled_modules:
                        continue
                    legacy["id"] = f"{namespace}.{module}"
                    changed += 1
        return self.model_copy(update={"values": values})

    def bind_data(self, data: TrainingData) -> "RLTrainingToml":
        if data.kind != "rl":
            raise ValueError("RL TOML requires RL data")
        values = deepcopy(self.values)
        orchestrator = values["orchestrator"]
        changed = 0
        for split, dataset in (
            ("train", data.train_path),
            ("eval", data.validation_path),
        ):
            section = orchestrator.get(split)
            if not isinstance(section, dict):
                raise ValueError(f"RL TOML needs [orchestrator.{split}] data")
            changed += _bind_rl_dataset(section.get("source"), dataset, legacy=True, extra=data.environment_args)
            changed += _bind_rl_dataset(section.get("env"), dataset, legacy=False, extra=data.environment_args)
        if not changed:
            raise ValueError("RL TOML has no supported train/eval source or environment tables")
        return self.model_copy(update={"values": values})


def _bind_rl_dataset(
    entries: object,
    dataset: str,
    *,
    legacy: bool,
    extra: dict[str, Any],
) -> int:
    if not isinstance(entries, list) or not entries:
        return 0
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("RL source/environment entries must be TOML tables")
        arguments = (
            entry.setdefault("legacy", {}).setdefault("args", {})
            if legacy
            else entry.setdefault("args", {})
        )
        if not isinstance(arguments, dict):
            raise ValueError("RL source/environment arguments must be TOML tables")
        arguments.update(extra)
        arguments["dataset"] = dataset
    return len(entries)


def load_training_toml(
    path: str | Path, *, kind: TrainingKind
) -> SFTTrainingToml | RLTrainingToml:
    return SFTTrainingToml.from_path(path) if kind == "sft" else RLTrainingToml.from_path(path)


def render_training_toml(
    kind: TrainingKind,
    *,
    template_toml_path: str | Path,
    data: TrainingData | dict[str, Any],
    output_toml_path: str | Path,
) -> Path:
    training_data = data if isinstance(data, TrainingData) else TrainingData.model_validate(data)
    if training_data.kind != kind:
        raise ValueError(
            f"data kind {training_data.kind!r} does not match requested TOML kind {kind!r}"
        )
    return load_training_toml(template_toml_path, kind=kind).bind_data(training_data).write(
        output_toml_path
    )
