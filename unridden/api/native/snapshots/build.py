"""Build the state-snapshot worker bundle against a layer-range base runtime.

The base must come from `scripts/unridden/build_base_runtime.py --patch
unridden/api/native/patches/gemma4-layer-range.patch`; its manifest records
the patch by sha256 and this build refuses any other base. The bundle has the
v1 layout (`build/<binary>`, `runtime/`, `build.json`) so the same install
checks apply, with `execution_mode` naming the default split profile and
`profiles` every profile the binary serves (ADR 0009).

Example:

    python -m unridden.api.native.snapshots.build \\
        --base build/llama-base-v0.4.1-split --output build/snapshot-worker
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from unridden.api.native.build import (
    TESTED_LLAMA_REVISION,
    TESTED_LLAMA_TAG,
    Row,
    bundle_digest,
    digest,
    sha256_helper,
    stage_runtime,
    validate_base,
)
from unridden.api.native.bundle import MANIFEST_NAME, RUNTIME_RELATIVE

PATCH_NAME = "gemma4-layer-range.patch"
BINARY_NAME = "unridden-snapshot-worker"
EXECUTABLE_RELATIVE = Path("build") / BINARY_NAME
PROFILE = "split18-30-v1"
FULL_PROFILE = "full-v1"
# Every profile the binary serves (`--profile`) and its execution mode.
PROFILES = {PROFILE: "split18-30", FULL_PROFILE: "full"}
PROTOCOL = "unridden-snapshot-v1"


def patch_path() -> Path:
    return Path(__file__).resolve().parent.parent / "patches" / PATCH_NAME


def require_layer_range_base(base_manifest: Row) -> str:
    patches = base_manifest.get("patches")
    expected = digest(patch_path())
    if not isinstance(patches, dict) or patches.get(PATCH_NAME) != expected:
        raise ValueError(
            f"Base runtime was not built with {PATCH_NAME} (sha256 {expected})"
        )
    if len(patches) != 1:
        raise ValueError("Base runtime carries patches other than the layer range")
    return expected


def build(base: Path, output: Path) -> Row:
    base = base.resolve()
    output = output.resolve()
    if output.exists():
        raise ValueError("Build output exists; choose a new directory")
    base_manifest, headers, libraries = validate_base(base)
    patch_sha256 = require_layer_range_base(base_manifest)
    native = Path(__file__).resolve().parent.parent
    cmake_dir = output / "cmake"
    subprocess.run(
        [
            "cmake",
            "-S",
            str(native),
            "-B",
            str(cmake_dir),
            f"-DLLAMA_SOURCE={headers}",
            f"-DLLAMA_BUILD={base / 'runtime'}",
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        check=True,
    )
    subprocess.run(
        ["cmake", "--build", str(cmake_dir), "--target", BINARY_NAME, "-j", "4"],
        check=True,
    )
    runtime_checksums = dict(base_manifest["runtime_sha256"])
    stage_runtime(libraries, output / RUNTIME_RELATIVE, runtime_checksums)
    executable = output / EXECUTABLE_RELATIVE
    executable.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cmake_dir / BINARY_NAME, executable)
    shutil.rmtree(cmake_dir)
    source_files = [
        native / "CMakeLists.txt",
        native / "worker-utils.h",
        native / "snapshots" / "snapshot-worker.cpp",
        native / "snapshots" / "build.py",
        patch_path(),
    ]
    sha_source = sha256_helper(headers)
    revision = str(base_manifest["llama_revision"])
    manifest: Row = {
        "schema_version": 2,
        "llama_revision": revision,
        "llama_ref": str(base_manifest.get("llama_ref", revision)),
        "tested_llama_tag": TESTED_LLAMA_TAG,
        "tested_llama_revision": TESTED_LLAMA_REVISION,
        "tested_revision": revision == TESTED_LLAMA_REVISION,
        "base_build_json_sha256": digest(base / "build.json"),
        "patches": {PATCH_NAME: patch_sha256},
        "cuda": bool(base_manifest.get("cuda", False)),
        "cuda_architectures": base_manifest.get("cuda_architectures"),
        "cuda_redistributables": base_manifest.get("cuda_redistributables", []),
        "executable": str(EXECUTABLE_RELATIVE),
        "runtime_dir": str(RUNTIME_RELATIVE),
        "runtime_sha256": runtime_checksums,
        "runtime_bundle_sha256": bundle_digest(runtime_checksums),
        "source_sha256": {path.name: digest(path) for path in source_files},
        "helper_sha256": {
            "sha256.c": digest(sha_source),
            "sha256.h": digest(sha_source.with_suffix(".h")),
        },
        "executable_sha256": digest(executable),
        "runtime_config": {
            "context": 2048,
            "batch": 256,
            "ubatch": 256,
            "threads": 8,
            "attention": "causal",
            "swa_full": True,
            "cuda_fusion": True,
            "cuda_graphs": True,
            "layer_ranges": {"lower": [0, 18], "upper": [18, 30]},
            "profile_layer_ranges": {
                PROFILE: {"lower": [0, 18], "upper": [18, 30]},
                FULL_PROFILE: {"lower": [0, -1]},
            },
        },
        "protocol": PROTOCOL,
        # The default profile; `profiles` lists every one the binary serves.
        "profile": PROFILE,
        "profiles": PROFILES,
        "callbacks_enabled": False,
        "generated_tokens": 0,
        "execution_mode": "split18-30",
    }
    (output / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        manifest = build(args.base, args.output)
    except (TypeError, ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
