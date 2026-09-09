"""Stable W&B identities for training launch previews and receipts."""

from __future__ import annotations

from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, Field, computed_field


class WandbRunReference(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str = Field(min_length=1)
    project: str = Field(min_length=1)
    entity: str | None = None
    name: str = Field(min_length=1)
    offline: bool = False

    @computed_field
    @property
    def url(self) -> str | None:
        if self.offline or not self.entity:
            return None
        return (
            "https://wandb.ai/"
            f"{quote(self.entity, safe='')}/{quote(self.project, safe='')}/runs/"
            f"{quote(self.run_id, safe='')}"
        )
