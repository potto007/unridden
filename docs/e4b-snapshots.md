# Gemma 4 E4B snapshots: current source only

For a first decision, use the [prebuilt E4B quick start](../README.md#quick-start).
This guide adds the experimental `/v2` API: create a reusable context snapshot,
then ask questions against it. E4B uses **`full-v1` at checkpoint 42**. It has
no intermediate checkpoint or promotion.

**Release boundary, checked September 27, 2026:** the latest release is v0.4.0.
Its prebuilt `/v1` worker accepts E4B, but its snapshot worker only supports
`split18-30-v1` for 26B-A4B. Fetching that snapshot bundle will not enable E4B.
Use the current repository source and build the snapshot worker below; an
installed bundle must explicitly list `full-v1` in its manifest's `profiles`.
See [ADR 0009](decisions/0009-full-depth-snapshot-profile.md) and the
[qualification results](results/full-depth-snapshots.md).

## Install and build

Use Linux or WSL2, Python 3.12+, uv, Git, CMake, a C++ toolchain, and the CUDA
toolkit for your NVIDIA GPU. The qualified setup used an RTX 5090; a minimum
E4B GPU size has not been established. The model download is 5.13 GB, and the
build needs additional disk for llama.cpp and its runtime. No model is loaded
during the build.

From a current clone of this repository, run:

```bash
uv sync
uv run --with huggingface_hub hf download unsloth/gemma-4-E4B-it-GGUF \
  gemma-4-E4B-it-UD-Q4_K_XL.gguf --local-dir models

# Use your GPU's compute capability: 120 for an RTX 5090, 89 for an RTX 4090.
# Both output directories must be new; choose new names if they already exist.
uv run python scripts/unridden/build_base_runtime.py \
  --out build/llama-base-e4b-full --cuda --cuda-architectures 120 \
  --patch unridden/api/native/patches/gemma4-layer-range.patch
uv run python -m unridden.api.native.snapshots.build \
  --base build/llama-base-e4b-full --output build/api-snapshot-worker-e4b-full
```

The base runtime uses the tested llama.cpp release plus the layer-range patch
required by the snapshot worker build. E4B itself runs all 42 blocks through
the stock whole-model graph. Keep the worker and its own `build.json` together;
do not pair this binary with a manifest from a downloaded or older bundle.

Check the profile before loading the model:

```bash
uv run python - <<'PY'
import json
from pathlib import Path

manifest = json.loads(Path("build/api-snapshot-worker-e4b-full/build.json").read_text())
assert manifest.get("profiles", {}).get("full-v1") == "full", "Rebuild from current source"
print("E4B full-v1 profile is available")
PY
```

## Start the server

```bash
uv run python -m unridden.api.cli serve --gpu --snapshots --no-v1 \
  --snapshot-profile full-v1 \
  --model-path models/gemma-4-E4B-it-UD-Q4_K_XL.gguf \
  --snapshot-worker build/api-snapshot-worker-e4b-full/build/unridden-snapshot-worker \
  --snapshot-manifest build/api-snapshot-worker-e4b-full/build.json \
  --snapshot-store-dir build/snapshots-e4b-full \
  --host 127.0.0.1 --port 8091
```

`--no-v1` loads only the snapshot backend, so the service does not load a second
copy of the model. Use a separate store directory for each model and profile;
an existing store from another configuration is rejected. The server stays in
the foreground; wait for startup to complete before sending requests.

## Get a first answer

In a second terminal at the repository root, run this example. It checks the
advertised profile, creates a context snapshot from the bundled hello message,
and asks the same three questions through `/v2/decisions`:

```bash
uv run python - <<'PY'
import json
from pathlib import Path

import httpx

hello = json.loads(Path("unridden/examples/hello.json").read_text())
with httpx.Client(base_url="http://127.0.0.1:8091", timeout=120) as client:
    response = client.get("/v2/models")
    response.raise_for_status()
    model = response.json()["models"][0]
    assert model["profile"] == "full-v1", "Expected the E4B full-v1 server"
    assert model["limits"]["checkpoints"] == [42], "Expected E4B's 42 blocks"

    response = client.post("/v2/snapshots", json={
        "input": {"kind": "context", "state": hello["state"]},
        "checkpoints": [42],
        "ttl_seconds": 3600,
    })
    response.raise_for_status()
    snapshot_id = response.json()["snapshots"][0]["id"]

    response = client.post("/v2/decisions", json={
        "snapshot": {"id": snapshot_id, "relationship": "followup"},
        "questions": hello["questions"],
        "readout": {"completed_blocks": 42},
    })
    response.raise_for_status()
    print(json.dumps(response.json(), indent=2))
PY
```

The response contains `route`, `urgency`, and `duplicate_charge` answers,
`usage.output_tokens: 0`, and snapshot usage with profile `full-v1`. Reuse the
snapshot id for more questions about that state until it expires. This example
uses in-memory persistence with a one-hour lifetime; see `/docs` on the
running server for the request schema and persistence options.

The published E4B tests verify exact identity after snapshot restore. They do
not promise identical answers between `/v1` and `/v2`: token chunking can move
probabilities, especially on uncertain questions. The measured Tetris results
and limitations are in the [qualification report](results/full-depth-snapshots.md).
