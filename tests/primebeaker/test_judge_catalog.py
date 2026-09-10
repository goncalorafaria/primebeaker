from __future__ import annotations

from primebeaker.judge_catalog import (
    JudgeCatalogCLI,
    judge_profile_catalog_dir,
    list_judge_model_profiles,
    load_judge_model_profile,
)


QUOKKA = (
    "/weka/gfaria/prime_sft/outputs/"
    "quokka-sft-qwen35-9b-rltracer-xmlv1-search25-baseformula-ba8d07-step400/"
    "weights/step_400"
)


def test_packaged_judge_catalog_resolves_quokka_profile_and_template() -> None:
    profile = load_judge_model_profile(QUOKKA)

    assert profile.source_path.parent == judge_profile_catalog_dir()
    assert profile.source_path.name.startswith("quokka_sft_qwen35_9b")
    assert profile.prompt_template_path.name == "jtc-io-terminal-release-browse-xml-v1.json"
    assert profile.prompt_template_path.is_file()
    assert profile.settings["local_search_model_path"] == "localsearch:bc-v2-72k"


def test_judge_catalog_cli_lists_packaged_profiles() -> None:
    catalog = JudgeCatalogCLI()

    assert catalog.path() == str(judge_profile_catalog_dir())
    assert QUOKKA in {profile.model for profile in list_judge_model_profiles()}
    assert catalog.show(QUOKKA)["model"] == QUOKKA
