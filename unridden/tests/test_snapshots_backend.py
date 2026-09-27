"""Native stdio lifecycle tests for the snapshot worker transport.

A disposable CPU-only Python child speaks the JSONL protocol so the transport's
discipline can be tested without a model: correlation, reaping on ambiguous
failure, and typed worker errors that leave the child running.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from pathlib import Path

import pytest

from unridden.api.native.build import TESTED_LLAMA_REVISION
from unridden.api.snapshots.backend import SnapshotNativeBackend
from unridden.api.snapshots.errors import (
    SnapshotExecutionError,
    SnapshotNotFound,
    SnapshotProtocolError,
    SnapshotRequestError,
    SnapshotUnavailableError,
)

WORKER_SOURCE = r"""#!/usr/bin/env python3
import argparse, hashlib, json, os, sys, time
p=argparse.ArgumentParser()
p.add_argument('--model'); p.add_argument('--model-sha256')
p.add_argument('--runtime-sha256'); p.add_argument('--runtime-dir', required=True)
p.add_argument('--context', type=int); p.add_argument('--batch', type=int)
p.add_argument('--ubatch', type=int); p.add_argument('--threads', type=int)
p.add_argument('--gpu', action='store_true')
p.add_argument('--reference', action='store_true')
p.add_argument('--profile', default='split18-30-v1')
a=p.parse_args()
if not os.path.isdir(a.runtime_dir): sys.exit('runtime-dir is not a directory')
labels=['A','B','C','D']; tok=[11,12,13,14]
hello={'type':'hello','protocol':'unridden-snapshot-v1','profile':'split18-30-v1',
 'model_id':'local-gemma-unridden-v1','model_name':'fixture',
 'model_sha256':a.model_sha256,'runtime_sha256':a.runtime_sha256,'labels':labels,
 'label_token_ids':tok,'context_size':a.context,'batch_size':a.batch,
 'ubatch_size':a.ubatch,'threads':a.threads,'n_layer':30,'n_embd':2816,
 'split_block':18,'reference_context':a.reference,
 'context_prompt_version':'unridden-gemma-context-v1','generated_tokens':0,
 'callbacks_enabled':False}
if 'MODE_BADHELLO' in __file__: hello['n_layer']=28
if a.profile=='full-v1':
 hello.update(profile='full-v1', n_layer=42, n_embd=2560, split_block=None)
print(json.dumps(hello), flush=True)
print('snapshot fixture diagnostic', file=sys.stderr, flush=True)
def err(rid, code, reason=None):
 return {'type':'error','id':rid,'code':code,'reason':reason,'message':code}
for line in sys.stdin:
 req=json.loads(line); rid=req['id']; t=req['type']
 if 'MODE_HANG' in __file__: time.sleep(60)
 if 'MODE_CORRUPT' in __file__: print('{', flush=True); continue
 if 'MODE_BADID' in __file__:
  print(json.dumps({'type':'created','id':'nope','prompt_sha256':'a'*64,'tokens':10,
   'snapshots':[],'readout':None,'block_tokens':{'lower':0,'upper':0},
   'timing_ms':{'lower':0.0,'upper':0.0,'total':0.0},
   'generated_tokens':0}), flush=True); continue
 if t=='create':
  if req.get('labels')==['BUDGET']:
   print(json.dumps(err(rid,'invalid_request','budget')), flush=True); continue
  cps=req['checkpoints']; tokens=10; snaps=[]
  id18=cps.get('18'); id30=cps.get('30'); paired=bool(id18 and id30)
  kind='context' if req['freeze'].get('kind')=='context' else 'readout'
  if id18: snaps.append({'snapshot_id':id18,'completed_blocks':18,'parent':None,
   'tokens':tokens,'bytes':{'lower_kv':tokens,'upper_kv':0,'h18':tokens,'h30':0},
   'kind':kind,'prompt_sha256':'a'*64})
  if id30: snaps.append({'snapshot_id':id30,'completed_blocks':30,
   'parent':id18 if paired else None,'tokens':tokens,
   'bytes':{'lower_kv':0 if paired else tokens,'upper_kv':tokens,'h18':0,'h30':tokens},
   'kind':kind,'prompt_sha256':'a'*64})
  readout=None
  if req.get('labels') and id30:
   n=len(req['labels']); readout={'label_logits':[float(i) for i in range(n)],
    'label_token_ids':tok[:n],'allowed_label_mass':0.5,
    'full_vocabulary_argmax':{'token_id':11,'logit':5.0}}
  print(json.dumps({'type':'created','id':rid,'prompt_sha256':'a'*64,'tokens':tokens,
   'snapshots':snaps,'readout':readout,
   'block_tokens':{'lower':tokens,'upper':tokens if id30 else 0},
   'timing_ms':{'lower':1.0,'upper':1.0,'total':3.0},
   'generated_tokens':0}), flush=True); continue
 if t=='inspect':
  if req['snapshot_id']=='unknown':
   print(json.dumps(err(rid,'snapshot_not_found')), flush=True); continue
  print(json.dumps({'type':'snapshot','id':rid,'snapshot_id':req['snapshot_id'],
   'completed_blocks':30,'parent':None,'tokens':10,
   'bytes':{'lower_kv':10,'upper_kv':10,'h18':0,'h30':10},'kind':'context',
   'prompt_sha256':'a'*64,'token_ids':[1,2,3],'resident':True,
   'holds':{'h18':False,'h30':True,'last_normalized':True,'upper_kv':True}}),
   flush=True); continue
 if t=='evaluate':
  print(json.dumps(err(rid,'execution_error')), flush=True); continue
 if t=='save':
  fn=os.path.join(req['directory'],'lower_kv.bin'); payload=b'kv'
  open(fn,'wb').write(payload); sha=hashlib.sha256(payload).hexdigest()
  print(json.dumps({'type':'saved','id':rid,'snapshot_id':req['snapshot_id'],
   'files':{'lower_kv.bin':{'bytes':len(payload),'sha256':sha}}}), flush=True); continue
 if t=='load':
  print(json.dumps({'type':'loaded','id':rid,'snapshot_id':req['snapshot_id'],
   'completed_blocks':30,'parent':None,'tokens':10,
   'bytes':{'lower_kv':10,'upper_kv':10,'h18':0,'h30':10},
   'kind':'context','prompt_sha256':'a'*64}), flush=True); continue
 if t=='drop':
  print(json.dumps({'type':'dropped','id':rid,'snapshot_id':req['snapshot_id'],
   'freed_bytes':100}), flush=True); continue
 print(json.dumps(err(rid,'execution_error')), flush=True)
"""


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _install(
    tmp_path: Path, mode: str = "normal", profiles: dict[str, str] | None = None
) -> tuple[Path, Path, Path]:
    worker = tmp_path / "worker"
    worker.write_text(WORKER_SOURCE)
    if mode != "normal":
        renamed = tmp_path / f"worker-MODE_{mode.upper()}"
        worker.rename(renamed)
        worker = renamed
    worker.chmod(0o755)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"fake model")
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    library = runtime / "libfixture.so"
    library.write_bytes(b"runtime")
    checksums = {library.name: _digest(library)}
    manifest = tmp_path / "build.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "llama_revision": TESTED_LLAMA_REVISION,
                "runtime_dir": runtime.name,
                "runtime_sha256": checksums,
                "runtime_bundle_sha256": hashlib.sha256(
                    json.dumps(
                        checksums, sort_keys=True, separators=(",", ":")
                    ).encode()
                ).hexdigest(),
                "executable": worker.name,
                "executable_sha256": _digest(worker),
                "protocol": "unridden-snapshot-v1",
                "profile": "split18-30-v1",
                "patches": {"gemma4-layer-range.patch": "0" * 64},
                "generated_tokens": 0,
                "callbacks_enabled": False,
                "execution_mode": "split18-30",
            }
            # A bundle from before ADR 0009 has no `profiles`.
            | ({"profiles": profiles} if profiles is not None else {})
        )
    )
    return worker, model, manifest


def _backend(tmp_path: Path, mode: str = "normal") -> SnapshotNativeBackend:
    worker, model, manifest = _install(tmp_path, mode)
    return SnapshotNativeBackend(
        worker_path=worker,
        model_path=model,
        manifest_path=manifest,
        startup_timeout=2.0,
        default_timeout=2.0,
    )


@pytest.mark.asyncio
async def test_handshake_returns_the_profile(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    profile = await backend.start()
    await backend.close()

    assert profile.protocol == "unridden-snapshot-v1"
    assert profile.profile == "split18-30-v1"
    assert profile.n_layer == 30 and profile.split_block == 18


@pytest.mark.asyncio
async def test_full_profile_starts_from_a_bundle_that_serves_it(
    tmp_path: Path,
) -> None:
    worker, model, manifest = _install(
        tmp_path, profiles={"split18-30-v1": "split18-30", "full-v1": "full"}
    )
    backend = SnapshotNativeBackend(
        worker_path=worker,
        model_path=model,
        manifest_path=manifest,
        snapshot_profile="full-v1",
        startup_timeout=2.0,
    )
    profile = await backend.start()
    await backend.close()

    assert profile.profile == "full-v1"
    assert profile.checkpoints == [42]


@pytest.mark.asyncio
async def test_full_profile_is_refused_by_a_split_only_bundle(tmp_path: Path) -> None:
    worker, model, manifest = _install(tmp_path)
    backend = SnapshotNativeBackend(
        worker_path=worker,
        model_path=model,
        manifest_path=manifest,
        snapshot_profile="full-v1",
        startup_timeout=2.0,
    )
    with pytest.raises(SnapshotUnavailableError, match="does not serve the full-v1"):
        await backend.start()
    assert backend.ready is False


@pytest.mark.asyncio
async def test_request_deadline_covers_a_worker_that_stops_reading_stdin(
    tmp_path: Path,
) -> None:
    worker, model, manifest = _install(tmp_path)
    worker.write_text(
        WORKER_SOURCE.replace(
            "for line in sys.stdin:",
            "while True: time.sleep(60)\nfor line in sys.stdin:",
        )
    )
    installed = json.loads(manifest.read_text())
    installed["executable_sha256"] = _digest(worker)
    manifest.write_text(json.dumps(installed))
    backend = SnapshotNativeBackend(
        worker_path=worker,
        model_path=model,
        manifest_path=manifest,
        default_timeout=0.1,
        startup_timeout=2.0,
    )
    await backend.start()
    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(
                backend.create(
                    messages=[{"role": "user", "content": "x" * (512 * 1024)}],
                    answer_prefix="",
                    freeze={"kind": "readout"},
                    checkpoints={"30": "snap_" + "1" * 32},
                    labels=None,
                    top_logits=0,
                    timeout=0.1,
                ),
                timeout=0.6,
            )
        assert time.monotonic() - started < 0.5
        assert backend.ready is False
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_create_round_trip(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    await backend.start()
    created = await backend.create(
        messages=[{"role": "user", "content": "state"}],
        answer_prefix="",
        freeze={"kind": "context", "content_bytes": 5},
        checkpoints={"18": "snap_" + "0" * 32, "30": "snap_" + "1" * 32},
        labels=None,
        top_logits=0,
    )
    await backend.close()

    blocks = {row.completed_blocks for row in created.snapshots}
    assert blocks == {18, 30}


@pytest.mark.asyncio
async def test_bad_hello_is_refused_and_reaped(tmp_path: Path) -> None:
    backend = _backend(tmp_path, "badhello")
    with pytest.raises(SnapshotUnavailableError):
        await backend.start()
    assert backend.pid is None
    assert backend.ready is False


@pytest.mark.asyncio
async def test_typed_worker_errors_leave_the_child_running(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    await backend.start()

    with pytest.raises(SnapshotNotFound):
        await backend.inspect(snapshot_id="unknown")
    assert backend.ready is True

    with pytest.raises(SnapshotExecutionError):
        await backend.evaluate(
            snapshot_id="snap_" + "2" * 32, readout_blocks=30, questions=[]
        )
    assert backend.ready is True

    with pytest.raises(SnapshotRequestError) as caught:
        await backend.create(
            messages=[{"role": "user", "content": "s"}],
            answer_prefix="",
            freeze={"kind": "readout"},
            checkpoints={"30": "snap_" + "3" * 32},
            labels=["BUDGET"],
            top_logits=0,
        )
    assert caught.value.reason == "budget"
    assert backend.ready is True
    await backend.close()


@pytest.mark.asyncio
async def test_corrupt_line_reaps_the_child(tmp_path: Path) -> None:
    backend = _backend(tmp_path, "corrupt")
    await backend.start()
    owned = backend.pid

    with pytest.raises(SnapshotProtocolError):
        await backend.inspect(snapshot_id="snap_" + "0" * 32)

    assert backend.ready is False
    assert backend.pid is None
    assert owned is not None
    with pytest.raises(ProcessLookupError):
        os.kill(owned, 0)


@pytest.mark.asyncio
async def test_correlation_mismatch_reaps_the_child(tmp_path: Path) -> None:
    backend = _backend(tmp_path, "badid")
    await backend.start()

    with pytest.raises(SnapshotProtocolError):
        await backend.create(
            messages=[{"role": "user", "content": "s"}],
            answer_prefix="",
            freeze={"kind": "context", "content_bytes": 1},
            checkpoints={"18": "snap_" + "0" * 32},
            labels=None,
            top_logits=0,
        )
    assert backend.ready is False


@pytest.mark.asyncio
async def test_hang_times_out_and_reaps(tmp_path: Path) -> None:
    backend = _backend(tmp_path, "hang")
    await backend.start()
    owned = backend.pid

    with pytest.raises(TimeoutError):
        await backend.inspect(snapshot_id="snap_" + "0" * 32, timeout=0.3)

    assert backend.ready is False
    assert owned is not None
    with pytest.raises(ProcessLookupError):
        os.kill(owned, 0)


@pytest.mark.asyncio
async def test_save_load_drop_serve_the_store(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    await backend.start()
    directory = tmp_path / "blob"
    directory.mkdir()

    files = await backend.save("snap_" + "0" * 32, directory)
    assert "lower_kv.bin" in files
    await backend.load(
        "snap_" + "0" * 32, directory, {name: e.sha256 for name, e in files.items()}
    )
    freed = await backend.drop("snap_" + "0" * 32)
    await backend.close()

    assert freed == 100
