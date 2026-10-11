"""Execute the saved-analysis notebook in a portable checkout, with scoped guards.

Adapted from the historical source-intake nbclient runner. Installation happens
before this command; the kernel cannot use the network, weights or blocked roots.
This is a Python I/O guard, not an operating-system security sandbox.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import tempfile

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpec


GUARD = '''import json, os, socket, sys
from pathlib import Path
blocked = [Path(p).resolve() for p in json.loads(os.environ["ANALYSIS_BLOCKED_ROOTS"])]
def guard(event, args):
    if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto"}:
        raise RuntimeError("Network forbidden during saved analysis")
    if event == "subprocess.Popen":
        raise PermissionError("Child processes forbidden during saved analysis")
    if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
        path = Path(os.fsdecode(args[0])).resolve()
        if any(path.is_relative_to(p) for p in blocked):
            raise RuntimeError("Original research checkout forbidden")
        if "weights" in path.parts or path.suffix in {".safetensors", ".pt", ".ckpt"} or (path.suffix in {".bin", ".pth"} and "site-packages" not in path.parts):
            raise RuntimeError("Model weights forbidden")
sys.addaudithook(guard)
import hebrew_acronyms
package_path = Path(hebrew_acronyms.__file__).resolve()
portable_root = Path(os.environ["ANALYSIS_PORTABLE_ROOT"]).resolve()
if not package_path.is_relative_to(portable_root / "src"):
    raise AssertionError(f"Package is not installed from portable source: {package_path}")
probes = {"package_path": str(package_path), "package_from_portable_source": True}
with socket.socket() as sock:
    try:
        sock.connect(("127.0.0.1", 9))
    except RuntimeError:
        probes["network_probe_blocked"] = True
    else:
        raise AssertionError("Network guard inactive")
for i, parent in enumerate(blocked):
    try:
        (parent / "README.md").read_bytes()
    except RuntimeError:
        probes[f"source_probe_{i}_blocked"] = True
    else:
        raise AssertionError("Source guard inactive")
try:
    (Path.cwd() / "weights" / "probe.pt").read_bytes()
except RuntimeError:
    probes["weights_probe_blocked"] = True
else:
    raise AssertionError("Weights guard inactive")
Path(os.environ["ANALYSIS_GUARD_RECEIPT"]).write_text(json.dumps(probes))
'''


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--blocked-root', type=Path, action='append', required=True)
    parser.add_argument('--revision', required=True, help='Commit used to prepare the portable source')
    parser.add_argument('--receipt', type=Path, required=True, help='New receipt; existing files are never replaced')
    args = parser.parse_args()
    root = args.root.resolve()
    receipt_path = args.receipt.resolve()
    if receipt_path.exists():
        raise FileExistsError(receipt_path)
    if any(root.is_relative_to(p.resolve()) for p in args.blocked_root):
        raise ValueError('Portable source must be outside every blocked checkout')
    notebook = root / 'notebooks/experimental_study.ipynb'
    nb = nbformat.read(notebook, as_version=4)
    source_hash = hashlib.sha256(json.dumps([
        {'cell_type': c.cell_type, 'source': c.source} for c in nb.cells
    ], ensure_ascii=False).encode()).hexdigest()
    with tempfile.TemporaryDirectory(prefix='saved-analysis-') as directory:
        run = Path(directory)
        startup = run / 'ipython/profile_default/startup'
        startup.mkdir(parents=True)
        (startup / '00-offline.py').write_text(GUARD)
        km = KernelManager()
        km._kernel_spec = KernelSpec(
            argv=[sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
            display_name='Offline saved analysis', language='python')
        client = NotebookClient(nb, km=km, timeout=180, resources={'metadata': {'path': str(root)}})
        env = {k: v for k, v in os.environ.items()
               if not any(t in k.upper() for t in ('TOKEN', 'SECRET', 'API_KEY', 'PASSWORD'))}
        env.pop('PYTHONPATH', None)
        env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                   HF_HOME=str(run / 'empty-model-cache'),
                   MPLBACKEND='module://matplotlib_inline.backend_inline',
                   MPLCONFIGDIR=str(run / 'matplotlib'), IPYTHONDIR=str(run / 'ipython'),
                   PYTHONDONTWRITEBYTECODE='1',
                   ANALYSIS_BLOCKED_ROOTS=json.dumps([str(p.resolve()) for p in args.blocked_root]),
                   ANALYSIS_GUARD_RECEIPT=str(run / 'guard.json'), ANALYSIS_PORTABLE_ROOT=str(root))
        client.on_cell_start = lambda cell, cell_index, **kwargs: print(f'Cell {cell_index + 1}/{len(nb.cells)} ({cell.cell_type})', flush=True)
        client.execute(env=env)
        probes = json.loads((run / 'guard.json').read_text())
    nbformat.validate(nb)
    nbformat.write(nb, notebook)
    receipt = {
        'executed_at_utc': datetime.now(timezone.utc).isoformat(),
        'code_base_commit': args.revision,
        'python': platform.python_version(), 'platform': platform.platform(),
        'packages': dict(sorted((d.metadata['Name'], d.version) for d in importlib.metadata.distributions())),
        'execution': 'all cells top to bottom; saved analysis only; no development collection cells',
        'guard_scope': 'kernel Python audit hooks; network, child processes, model weights and specified original checkouts blocked',
        'blocked_checkout_names': [p.name for p in args.blocked_root],
        'guard_verification': probes,
        'cells': len(nb.cells),
        'executed_code_cells': sum(c.cell_type == 'code' and c.execution_count is not None for c in nb.cells),
        'errors': sum(o.output_type == 'error' for c in nb.cells for o in c.get('outputs', [])),
        'notebook_source_sha256': source_hash, 'executed_notebook_sha256': sha(notebook),
        'runner_sha256': sha(Path(__file__)),
        'analysis_code_sha256': {str(p.relative_to(root)): sha(p) for p in sorted((root / 'src').rglob('*.py'))},
        'evidence_inventory_sha256': sha(root / 'saved-results/files.json'),
    }
    with receipt_path.open('x') as handle:
        handle.write(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: receipt[k] for k in ('cells', 'executed_code_cells', 'errors', 'guard_verification')}, indent=2))


if __name__ == '__main__':
    main()
