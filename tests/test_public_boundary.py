"""Keep the user repository and the twelve open scenarios structurally complete."""
from pathlib import Path

import pytest
from agrel_public.scenarios.registry import BUILTIN, OPTIONAL, ORDER, SCENARIOS, configs


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "agrel_public"


def test_all_twelve_scenarios_and_both_configs_are_public():
    expected = {
        "drug_screen", "drug_finetune", "molecule_neighbors", "cytology",
        "ocr_cache", "genomics_splice", "medical_ultrasound", "materials_screen",
        "singlecell_neighbors", "biomed_rag", "llm_biomed_tables", "agent_tools",
    }
    assert set(ORDER) == expected
    assert len(BUILTIN) == 8 and len(OPTIONAL) == 4
    assert all(len(configs(name)) == 2 for name in ORDER)
    assert {SCENARIOS[name]["kind"] for name in ORDER} == {
        "supervised", "train", "retrieval", "cache", "llm"
    }


def test_scenario_runtime_sources_ship_in_user_repository():
    required = {
        "scenarios/registry.py", "scenarios/engine.py", "scenarios/llm.py",
        "scenarios/assets.py", "selection.py", "worker.py", "processes.py", "cli.py",
    }
    assert all((PACKAGE / relative).is_file() for relative in required)


def test_maintainer_material_does_not_ship_in_user_repository():
    if (ROOT.parent / "publisher/protected/stage.py").is_file():
        pytest.skip("Publisher tests inspect the staged public tree; canonical tree retains maintainer test records")
    forbidden_names = {"TESTED.md", "DO_NOT_PUBLISH.md", "RELEASE_GATE.json"}
    forbidden_parts = {"validation", "maintainer-only", "private_server", "publisher", "internal-dist", "private_extensions"}
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        assert path.name not in forbidden_names
        assert not forbidden_parts.intersection(relative.parts)
        assert not any(part.startswith("agrel_detector_core") for part in relative.parts)
        assert path.suffix.lower() not in {".so", ".pyd", ".dll", ".dylib"}
        assert not path.is_symlink()


def test_routine_numerical_sources_ship_in_user_repository():
    for name in ("exact.py", "reference.py", "frozen.py", "native_smid.py", "probes.py"):
        assert (PACKAGE / name).is_file()
    for name in ("smid_probe.cu", "build_native.py"):
        assert (PACKAGE / "assets/native/source" / name).is_file()
    assert (PACKAGE / "assets/native/smid_probe.fatbin").is_file()


def test_open_client_has_no_compulsory_private_core_dependency():
    project = (ROOT / "pyproject.toml").read_text()
    assert "computeproof-core" not in project
    assert "computeproof_core" not in project
