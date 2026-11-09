"""Execute the standalone deep-tilt notebook in an empty directory."""
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

HERE = Path(__file__).resolve().parent
target = HERE / "RollBack_SMC_deep_tilts_Colab.ipynb"
notebook = nbformat.read(target, as_version=4)
start = time.perf_counter()
with tempfile.TemporaryDirectory(prefix="deep_rollback_notebook_") as temporary:
    root = Path(temporary)
    scratch = root / "empty_working_directory"
    scratch.mkdir()
    kernel_dir = root / "kernels" / "python3"
    kernel_dir.mkdir(parents=True)
    (kernel_dir / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Python 3", "language": "python"}), encoding="utf-8")
    spec = KernelSpecManager(kernel_dirs=[str(kernel_dir.parent)], ensure_native_kernel=False)
    manager = KernelManager(kernel_name="python3", kernel_spec_manager=spec)
    client = NotebookClient(notebook, km=manager, timeout=7200, allow_errors=False,
                            resources={"metadata": {"path": str(scratch)}})
    def on_start(cell, cell_index):
        if cell.cell_type == "code":
            print(f"Executing cell {cell_index+1}/{len(notebook.cells)}", flush=True)
    def on_done(cell, cell_index, execute_reply=None):
        nbformat.write(notebook, HERE / "deep_execution_checkpoint.ipynb")
        for output in cell.get("outputs", []):
            if output.output_type == "stream":
                print(output.text[-1800:], flush=True)
    client.on_cell_start = on_start
    client.on_cell_executed = on_done
    print("Running standalone notebook from an empty directory.", flush=True)
    try:
        client.execute()
    finally:
        manager.shutdown_kernel(now=True)
    nbformat.validate(notebook)
    nbformat.write(notebook, target)
    shutil.copytree(scratch / "deep_tilt_results", HERE / "deep_results", dirs_exist_ok=True)
code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
assert all(cell.execution_count is not None for cell in code_cells)
assert not any(output.output_type == "error" for cell in code_cells for output in cell.outputs)
result = {"status": "success", "code_cells_executed": len(code_cells),
          "empty_working_directory": True, "errors": 0,
          "seconds": time.perf_counter()-start}
(HERE / "deep_execution_verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2), flush=True)
