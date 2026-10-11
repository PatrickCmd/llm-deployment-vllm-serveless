#!/usr/bin/env python3
"""
Setup: Verify Environment
Checks the lab environment and downloads the SmolLM-135M model.

Uses uv (https://docs.astral.sh/uv/) as the package manager and a
virtual environment at <project root>/.venv, where the project root is
the directory containing this script.

Run it with any Python interpreter:

    python3 verify_environment.py

The script installs uv if needed (macOS/Linux), creates the venv if it
does not exist, and re-executes itself inside the venv so all package
checks run against it.
"""

import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
VENV_DIR = PROJECT_ROOT / ".venv"
MARKERS_DIR = PROJECT_ROOT / "markers"
VENV_PYTHON = (
    VENV_DIR / "Scripts" / "python.exe"
    if platform.system() == "Windows"
    else VENV_DIR / "bin" / "python"
)
UV_INSTALL_URL = "https://astral.sh/uv/install.sh"
UV_BIN_DIRS = [Path.home() / ".local" / "bin", Path.home() / ".cargo" / "bin"]
REEXEC_GUARD = "VLLM_LAB_VENV_REEXEC"

# Keep our prints ordered with subprocess output and survive os.execve().
sys.stdout.reconfigure(line_buffering=True)


def _refresh_path():
    """Make freshly installed uv binaries discoverable in this process."""
    extra = [str(d) for d in UV_BIN_DIRS if d.is_dir()]
    current = os.environ.get("PATH", "")
    os.environ["PATH"] = os.pathsep.join(extra + [current]) if extra else current


def find_uv():
    """Return the path to the uv executable, or None."""
    _refresh_path()
    return shutil.which("uv")


def install_uv():
    """Install uv with the official installer. Supports macOS and Linux."""
    system = platform.system()
    if system not in ("Darwin", "Linux"):
        print(f"  FAIL - Automatic uv install is not supported on {system}")
        print("  Fix: Install uv manually: https://docs.astral.sh/uv/getting-started/installation/")
        return None

    if shutil.which("curl"):
        fetch = ["curl", "-LsSf", UV_INSTALL_URL]
    elif shutil.which("wget"):
        fetch = ["wget", "-qO-", UV_INSTALL_URL]
    else:
        print("  FAIL - Neither curl nor wget is available to download the uv installer")
        print(f"  Fix: Install curl, then run: curl -LsSf {UV_INSTALL_URL} | sh")
        return None

    print(f"  uv not found. Installing uv for {system} ({platform.machine()})...")
    try:
        script = subprocess.run(fetch, check=True, capture_output=True, text=True).stdout
        subprocess.run(["sh"], input=script, check=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"  FAIL - uv installer exited with status {e.returncode}")
        print(f"  Fix: Run manually: curl -LsSf {UV_INSTALL_URL} | sh")
        return None

    uv = find_uv()
    if uv is None:
        print("  FAIL - uv was installed but is not on PATH")
        print("  Fix: Add ~/.local/bin to your PATH and re-run this script")
    return uv


def ensure_uv():
    """Check 1: uv is installed (install it if missing)."""
    uv = find_uv()
    if uv:
        version = subprocess.run([uv, "--version"], capture_output=True, text=True).stdout.strip()
        print(f"  PASS - {version} at {uv}")
        return uv

    uv = install_uv()
    if uv:
        version = subprocess.run([uv, "--version"], capture_output=True, text=True).stdout.strip()
        print(f"  PASS - Installed {version} at {uv}")
    return uv


def ensure_venv(uv):
    """Check 2: a uv-managed venv exists at <project root>/.venv (create if missing)."""
    if VENV_PYTHON.exists():
        print(f"  PASS - Virtual environment found at {VENV_DIR}")
        return True

    print(f"  Virtual environment not found. Creating it at {VENV_DIR}...")
    try:
        subprocess.run([uv, "venv", str(VENV_DIR)], check=True, cwd=PROJECT_ROOT)
    except subprocess.CalledProcessError as e:
        print(f"  FAIL - 'uv venv' exited with status {e.returncode}")
        print(f"  Fix: Run 'uv venv {VENV_DIR}' manually and re-run this script")
        return False

    if not VENV_PYTHON.exists():
        print(f"  FAIL - Expected interpreter at {VENV_PYTHON} after 'uv venv'")
        return False

    print(f"  PASS - Virtual environment created at {VENV_DIR}")
    return True


def running_inside_venv():
    """True when the current interpreter belongs to the project venv."""
    try:
        return Path(sys.prefix).resolve() == VENV_DIR.resolve()
    except OSError:
        return False


def reexec_in_venv():
    """Replace this process with the same script run by the venv interpreter."""
    if os.environ.get(REEXEC_GUARD):
        print(f"  FAIL - Re-executed under {VENV_PYTHON} but still not inside the venv")
        print(f"  Fix: Run 'source {VENV_DIR}/bin/activate' and retry")
        return False

    print(f"  Re-running inside the virtual environment: {VENV_PYTHON}")
    env = dict(os.environ, **{REEXEC_GUARD: "1"})
    sys.stdout.flush()
    sys.stderr.flush()
    os.execve(str(VENV_PYTHON), [str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]], env)


def verify_environment():
    """Verify all lab prerequisites are met."""
    print("=" * 65)
    print("vLLM Explained Lab - Environment Verification")
    print("=" * 65)
    print(f"Project root: {PROJECT_ROOT}")

    checks_passed = 0
    checks_total = 6

    # Check 1: uv package manager
    print("\n[1/6] Checking uv package manager...")
    uv = ensure_uv()
    if not uv:
        return False
    checks_passed += 1

    # Check 2: Virtual environment in the project root
    print("\n[2/6] Checking Python virtual environment...")
    if not ensure_venv(uv):
        return False
    if not running_inside_venv():
        # execve does not return on success
        reexec_in_venv()
        return False
    print(f"  PASS - Running with {sys.executable}")
    checks_passed += 1

    # Check 3: Required packages
    print("\n[3/6] Checking required packages...")
    try:
        import torch
        import transformers
        print(f"  PASS - torch {torch.__version__}")
        print(f"  PASS - transformers {transformers.__version__}")
        checks_passed += 1
    except ImportError as e:
        print(f"  FAIL - Missing package: {e}")
        print("  Fix: Run 'uv pip install torch transformers'")
        return False

    # Check 4: vLLM (must be the CPU build)
    print("\n[4/6] Checking vLLM installation...")
    try:
        import vllm
        # Check the package metadata version, not vllm.__version__ -
        # the runtime __version__ strips the +cpu suffix ("0.24.0"),
        # while the metadata keeps it ("0.24.0+cpu"). vLLM's platform
        # detection reads the metadata version too.
        from importlib.metadata import version as pkg_version
        vllm_version = pkg_version("vllm")
        if "cpu" in vllm_version:
            print(f"  PASS - vllm {vllm_version} (CPU build)")
            checks_passed += 1
        else:
            print(f"  FAIL - vllm {vllm_version} is not the CPU build")
            print("  The generic PyPI wheel is a CUDA build and cannot run")
            print("  on this GPU-less machine (fails with 'Device string must")
            print("  not be empty' at engine init).")
            print("  Fix: Reinstall the CPU wheel:")
            print("    uv pip uninstall vllm")
            print("    uv pip install 'https://github.com/vllm-project/vllm/releases/download/v0.24.0/vllm-0.24.0+cpu-cp38-abi3-manylinux_2_34_x86_64.whl' --extra-index-url https://download.pytorch.org/whl/cpu")
            return False
    except ImportError as e:
        print(f"  FAIL - vLLM not installed: {e}")
        print("  Fix: Re-run the lab startup script or contact support")
        return False

    # Check 5: Additional packages
    print("\n[5/6] Checking additional packages...")
    try:
        import gradio
        import aiohttp
        import requests
        print(f"  PASS - gradio {gradio.__version__}")
        print(f"  PASS - aiohttp {aiohttp.__version__}")
        print(f"  PASS - requests {requests.__version__}")
        checks_passed += 1
    except ImportError as e:
        print(f"  FAIL - Missing package: {e}")
        print("  Fix: Run 'uv pip install gradio aiohttp requests'")
        return False

    # Check 6: Download SmolLM-135M model
    print("\n[6/6] Downloading SmolLM-135M model...")
    print("  This may take a minute on first run...")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = "HuggingFaceTB/SmolLM-135M"
        start_time = time.time()

        print(f"  Downloading tokenizer for {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        print(f"  Downloading model for {model_name}...")
        model = AutoModelForCausalLM.from_pretrained(model_name)

        elapsed = time.time() - start_time
        param_count = sum(p.numel() for p in model.parameters()) / 1e6

        print(f"  PASS - Model downloaded in {elapsed:.1f}s")
        print(f"  Model size: {param_count:.0f}M parameters")

        # Quick test generation
        print("\n  Running quick test generation...")
        inputs = tokenizer("Hello, world!", return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=10)
        test_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"  Test output: {test_output[:80]}...")
        print("  PASS - Model generates text successfully")

        # Clean up memory
        del model
        del tokenizer

        checks_passed += 1
    except Exception as e:
        print(f"  FAIL - Model download failed: {e}")
        return False

    # Summary
    print("\n" + "=" * 65)
    print(f"ENVIRONMENT CHECK: {checks_passed}/{checks_total} passed")
    print("=" * 65)

    if checks_passed == checks_total:
        print("\nAll checks passed! Your environment is ready.")
        print("\nLab Scenario:")
        print("  You are an ML engineer at InferenceIO.")
        print("  Mission: Use vLLM to serve SmolLM to concurrent users.")
        print("\nNext step: Run Task 1")
        print(f"  uv run python {PROJECT_ROOT / 'task_1_hf_baseline.py'}")

        # Create marker
        MARKERS_DIR.mkdir(parents=True, exist_ok=True)
        with open(MARKERS_DIR / "environment_verified.txt", "w") as f:
            f.write("ENVIRONMENT_VERIFIED\n")

        print("\nEnvironment verification complete!")
        return True
    else:
        print(f"\n{checks_total - checks_passed} check(s) failed. Fix the issues above and retry.")
        return False


if __name__ == "__main__":
    success = verify_environment()
    sys.exit(0 if success else 1)
