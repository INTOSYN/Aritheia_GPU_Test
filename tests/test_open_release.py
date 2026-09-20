"""Software-only tests for the open routine client and migration safety."""
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest
import agrel_public


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src/agrel_public"


@pytest.mark.parametrize("filename", ["exact.cpython-312-x86_64-linux-gnu.so", "probes.cp312-win_amd64.pyd", "frozen.abi3.so"])
def test_legacy_private_core_cannot_shadow_open_code(tmp_path, filename):
    (tmp_path / filename).write_bytes(b"not a real extension")
    with pytest.raises(ImportError, match="clean virtual environment"):
        agrel_public._reject_legacy_core(tmp_path)


def test_pure_source_and_separate_deep_extension_do_not_trigger_legacy_guard(tmp_path):
    (tmp_path / "exact.py").write_text("# ordinary open source\n")
    (tmp_path / "consented_deep.pyd").write_bytes(b"fixture")
    agrel_public._reject_legacy_core(tmp_path)


def test_actual_package_import_rejects_legacy_binary_before_loading_it(tmp_path):
    root = tmp_path / "agrel_public"
    root.mkdir()
    (root / "__init__.py").write_bytes((PACKAGE / "__init__.py").read_bytes())
    (root / "exact.so").write_bytes(b"fixture; must never be loaded")
    result = subprocess.run([sys.executable, "-B", "-c", "import agrel_public"], cwd=tmp_path,
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "Legacy computeproof-core" in result.stderr
    assert "reinstall the open ComputeProof client" in result.stderr


def test_installers_do_not_require_private_core_or_rename_python_abi():
    linux = (ROOT / "install-linux.sh").read_text()
    windows = (ROOT / "install-windows.ps1").read_text()
    for text in (linux, windows):
        assert "CORE_WHEEL" not in text
        assert "cp312-cp312" not in text
        assert "venv-rc6" in text
        assert "py3-none-any.whl" in text
    assert "Get-FileHash" in windows
    assert "[switch]$Run" in windows
    assert "--accept-download" not in windows


def test_native_developer_build_finds_windows_disassembler_without_running_cuda():
    path = PACKAGE / "assets/native/source/build_native.py"
    spec = importlib.util.spec_from_file_location("native_build_open_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.disassembler_path("/cuda/bin/nvcc") == "/cuda/bin/nvdisasm"
    assert module.disassembler_path("C:/CUDA/bin/nvcc.exe") == "C:/CUDA/bin/nvdisasm.exe"


def test_native_reference_kernel_binding_is_still_present():
    source = (PACKAGE / "native_smid.py").read_text()
    assert 'doc["fatbin_sha256"] != manifest()["fatbin_sha256"]' in source
    assert "Native reference/kernel contract mismatch" in source
