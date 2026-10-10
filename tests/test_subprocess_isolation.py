"""A cross-environment subprocess must not inherit the caller's import paths.

aggressor-wrappers and amylodeep live in separate conda environments here. The
runner launched the child with no ``env=`` at all, so the child's Python -- from
amylodeep's env -- still saw PYTHONPATH pointing at the CALLER's site-packages and
DYLD_LIBRARY_PATH at the caller's dylibs. A compiled extension built against one
environment then loads inside an interpreter built against another, which on macOS
is a SIGSEGV rather than an ImportError.

That matters most for this tool precisely because it is already the fragile case:
AmyloDeep loads PyTorch and JAX in one process, each with its own OpenMP runtime.
Two copies of that pair, from two environments, is worse than one.
"""

from __future__ import annotations

import os

import pytest

from aggressor_wrappers.runners.amylodeep import (
    _LEAKING_VARIABLES,
    AmyloDeepRunner,
    child_environment,
)

CALLER_ENV = {
    "PATH": "/opt/aw-env/bin:/usr/bin:/bin",
    "PYTHONPATH": "/opt/aw-env/lib/python3.12/site-packages",
    "PYTHONHOME": "/opt/aw-env",
    "PYTHONEXECUTABLE": "/opt/aw-env/bin/python",
    "DYLD_LIBRARY_PATH": "/opt/aw-env/lib",
    "DYLD_INSERT_LIBRARIES": "/opt/aw-env/lib/libsomething.dylib",
    "LD_LIBRARY_PATH": "/opt/aw-env/lib",
    "LD_PRELOAD": "/opt/aw-env/lib/libgomp.so",
    "VIRTUAL_ENV": "/opt/aw-env",
    "CONDA_PREFIX": "/opt/aw-env",
    "HF_HOME": "/data/hf",
    "AMYLODEEP_MODEL_DIR": "/data/amylodeep_models",
    "LANG": "en_US.UTF-8",
    "HTTPS_PROXY": "http://proxy:3128",
}
CHILD_PYTHON = "/opt/envs/amylodeep/bin/python"


@pytest.mark.parametrize("variable", sorted(_LEAKING_VARIABLES))
def test_every_import_path_variable_is_dropped(variable):
    clean = child_environment(CHILD_PYTHON, base=CALLER_ENV)
    assert variable not in clean, (
        f"{variable} leaks the caller's environment into the child, which is how "
        f"one env's compiled extensions end up in another env's interpreter"
    )


@pytest.mark.parametrize(
    "variable", ["HF_HOME", "AMYLODEEP_MODEL_DIR", "LANG", "HTTPS_PROXY"]
)
def test_variables_that_matter_are_passed_through(variable):
    """Sanitising must not become a reset: the model directory, the HF cache, the
    proxy and the locale all matter and none of them cross an ABI boundary."""
    clean = child_environment(CHILD_PYTHON, base=CALLER_ENV)
    assert clean[variable] == CALLER_ENV[variable]


def test_conda_prefix_and_path_point_at_the_child_environment():
    clean = child_environment(CHILD_PYTHON, base=CALLER_ENV)
    assert clean["CONDA_PREFIX"] == "/opt/envs/amylodeep"
    assert clean["PATH"].split(os.pathsep)[0] == "/opt/envs/amylodeep/bin"
    assert "/usr/bin" in clean["PATH"], "system tools must remain reachable"


def test_the_callers_bin_is_not_duplicated_in_path():
    clean = child_environment("/opt/aw-env/bin/python", base=CALLER_ENV)
    entries = clean["PATH"].split(os.pathsep)
    assert entries.count("/opt/aw-env/bin") == 1


def test_a_bare_interpreter_name_leaves_path_alone():
    """`python3` off PATH has no env prefix to derive, and inventing one would be
    worse than leaving PATH as it was."""
    clean = child_environment("python3", base=CALLER_ENV)
    assert clean["PATH"] == CALLER_ENV["PATH"]
    assert clean["CONDA_PREFIX"] == CALLER_ENV["CONDA_PREFIX"]


def test_kmp_duplicate_lib_ok_is_opt_in():
    assert "KMP_DUPLICATE_LIB_OK" not in child_environment(
        CHILD_PYTHON, base=CALLER_ENV
    )
    diagnostic = child_environment(
        CHILD_PYTHON, base=CALLER_ENV, kmp_duplicate_lib_ok=True
    )
    assert diagnostic["KMP_DUPLICATE_LIB_OK"] == "TRUE"


def test_sanitising_can_be_turned_off_to_compare():
    """Keeping the old behaviour available is what makes the fix falsifiable: if
    the crash survives sanitising, the environments were not the cause."""
    assert AmyloDeepRunner().sanitise_environment is True
    assert AmyloDeepRunner(sanitise_environment=False).sanitise_environment is False


def test_the_child_environment_does_not_mutate_the_parents():
    before = dict(os.environ)
    child_environment(CHILD_PYTHON)
    assert dict(os.environ) == before
