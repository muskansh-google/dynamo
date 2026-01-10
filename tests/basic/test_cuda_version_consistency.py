# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration test to check CUDA major version consistency across various packages.

This test ensures all CUDA-related components use the same major version (10+).
Checks: driver (nvidia-smi), toolkit (nvcc), environment variables, system packages (dpkg), and Python packages (pip).
"""

import re
import subprocess

import pytest

# Mark this with every framework to test every container
pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.integration,
    pytest.mark.parallel,
    pytest.mark.post_merge,
    pytest.mark.pre_merge,
    pytest.mark.sglang,
    pytest.mark.trtllm,
    pytest.mark.vllm,
]

# Easy to edit later:
IGNORE_PIP_PREFIXES = ("cupy", "nixl")


def sh(cmd: str) -> str:
    """
    Run command and return stdout only.
    We intentionally drop stderr to avoid noisy tools (pip warnings, etc.).
    """
    p = subprocess.run(
        ["bash", "-lc", f"{cmd} 2>/dev/null"],
        stdout=subprocess.PIPE,
        text=True,
        check=False,
    )
    return (p.stdout or "").strip()


def major_from_text(text: str) -> int | None:
    """Extract CUDA major version (10+) from arbitrary text; otherwise None."""
    if not text:
        return None

    # fmt: off
    pats = [
        r"\bCUDA_VERSION=([1-9]\d)\.",          # CUDA_VERSION=10.0, 11.0, 12.0, etc. (10-99)
        r"\bNV_CUDA_.*?_VERSION=([1-9]\d)\.",   # NV_CUDA_CUDART_VERSION=10.0...
        r"\+cuda([1-9]\d)\.",                   # ...+cuda10.0
        r"\bcuda\s*>=\s*([1-9]\d)\.",           # cuda>=10.0 ...
        r"\brelease\s+([1-9]\d)\.",             # nvcc: release 10.0
        r"-([1-9]\d)-\d+\b",                    # dpkg: ...-10-0
        r"\bcuda([1-9]\d)x\b",                  # cupy-cuda10x (from name)
        r"[-+]cu([1-9])([0-9])\d?\b",           # -cu100 or +cu129 (CUDA 10-99) - capture first 2 digits separately
    ]
    # fmt: on
    for i, pat in enumerate(pats):
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            # Special handling for cu### pattern (last pattern)
            if i == len(pats) - 1 and len(m.groups()) >= 2:
                # cu129 -> major=12, cu200 -> major=20 (first two digits)
                maj = int(m.group(1) + m.group(2))
            else:
                maj = int(m.group(1))
            if maj >= 10:  # Support CUDA 10 and above
                return maj
    return None


def pip_cuda_major_from_line(line: str) -> int | None:
    """
    Given a pip freeze line like 'torch==2.9.0+cu130', infer CUDA major.
    Returns major version (10+) or None.
    """
    return major_from_text(line)


def keep_pip_line(line: str) -> bool:
    """
    Ignore some packages from pip signal (editable allow/ignore policy).
    """
    name = line.split("==", 1)[0].strip().lower()
    return not name.startswith(IGNORE_PIP_PREFIXES)


@pytest.mark.cuda
def test_cuda_major_consistency() -> None:
    """
    Collect CUDA major versions (10+) from predefined signals and assert consistency.
    Prints a readable report with full relevant output when failing.
    """

    signals = [
        ("env:CUDA_VERSION", "env | grep -i '^CUDA_VERSION='"),
        ("env:NV_CUDA_CUDART_VERSION", "env | grep -i '^NV_CUDA_CUDART_VERSION='"),
        ("env:NV_CUDA_LIB_VERSION", "env | grep -i '^NV_CUDA_LIB_VERSION='"),
        ("env:NV_LIBNCCL_PACKAGE", "env | grep -i '^NV_LIBNCCL_PACKAGE='"),
        ("env:NVIDIA_REQUIRE_CUDA", "env | grep -i '^NVIDIA_REQUIRE_CUDA='"),
        ("nvcc", "nvcc --version | grep -i 'release' || nvcc --version"),
        ("dpkg:cuda-*", "dpkg -l | grep -E '^(ii|hi)\\s+cuda-.*-[1-9][0-9]-'"),
        (
            "dpkg:libcublas/libnccl",
            "dpkg -l | grep -E '^(ii|hi)\\s+lib(cublas|nccl).*-[1-9][0-9]-'",
        ),
        # pip signal: gather a targeted list, then infer majors per line (excluding ignored prefixes)
        (
            "pip:selected",
            "python -m pip list --format=freeze | grep -Ei '(cuda|cudnn|nccl|nvshmem|\\+cu[1-9][0-9]|-cu[1-9][0-9]|^(torch|torchaudio|torchvision)==)'",
        ),
    ]

    rows: list[tuple[str, int | None, list[str]]] = []

    for label, cmd in signals:
        out = sh(cmd)
        lines = [ln.strip() for ln in out.splitlines() if ln.strip()]

        if label.startswith("pip:"):
            lines = [ln for ln in lines if keep_pip_line(ln)]
            majors = {pip_cuda_major_from_line(ln) for ln in lines}
            majors.discard(None)
            maj = majors.pop() if len(majors) == 1 else None  # None if ambiguous/mixed
            # If mixed, we’ll surface it via the global consistency check below.
        else:
            maj = major_from_text(out)

        rows.append((label, maj, lines if lines else ["<no output>"]))

    # Compute all detected majors across *all* signals, including per-line pip majors.
    detected: list[int] = []
    for label, maj, lines in rows:
        if label.startswith("pip:"):
            for ln in lines:
                m = pip_cuda_major_from_line(ln)
                if m is not None:
                    detected.append(m)
        else:
            if maj is not None:
                detected.append(maj)

    if not detected:
        pytest.skip("No CUDA major version (10+) detected from any signal.")

    unique = sorted(set(detected))

    # Build a readable multi-line report (no truncation to first line).
    report = [
        "CUDA major signals (pip ignores prefixes: "
        + ", ".join(IGNORE_PIP_PREFIXES)
        + "):"
    ]
    for label, maj, lines in rows:
        maj_s = str(maj) if maj is not None else "-"
        report.append(f"  {maj_s:>2}  {label}")
        for ln in lines[:50]:  # keep it readable; adjust if you want more
            report.append(f"      {ln}")
        if len(lines) > 50:
            report.append(f"      ... ({len(lines) - 50} more lines)")

    assert len(unique) == 1, (
        "\n".join(report) + f"\n\nInconsistent CUDA majors detected: {unique}"
    )
