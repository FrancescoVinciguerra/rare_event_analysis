"""Execute the delivered .ipynb through Jupyter in a fresh empty directory.

Execution tooling is local; the notebook itself contains no machine paths.
"""
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


def main():
    path = HERE / "RollBack_SMC_Colab.ipynb"
    notebook = nbformat.read(path, as_version=4)
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="rollback_notebook_") as temporary:
        root = Path(temporary)
        scratch = root / "empty_working_directory"
        scratch.mkdir()
        assert not list(scratch.iterdir())
        kernel_dir = root / "kernels" / "python3"
        kernel_dir.mkdir(parents=True)
        (kernel_dir / "kernel.json").write_text(json.dumps({"argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"], "display_name": "Python 3", "language": "python"}), encoding="utf-8")
        spec = KernelSpecManager(kernel_dirs=[str(kernel_dir.parent)], ensure_native_kernel=False)
        manager = KernelManager(kernel_name="python3", kernel_spec_manager=spec)
        client = NotebookClient(notebook, km=manager, timeout=1800, allow_errors=False,
                                resources={"metadata": {"path": str(scratch)}})
        def cell_start(cell, cell_index):
            if cell.cell_type == "code":
                print(f"Executing cell {cell_index+1}/{len(notebook.cells)}", flush=True)
        def cell_done(cell, cell_index, execute_reply=None):
            nbformat.write(notebook, HERE / "execution_checkpoint.ipynb")
            for output in cell.get("outputs", []):
                if output.output_type == "stream":
                    print(output.text[-3000:], flush=True)
        client.on_cell_start = cell_start
        client.on_cell_executed = cell_done
        print("Executing in a fresh empty directory, with no local modules.", flush=True)
        # A small progress marker outside the notebook allows monitoring saved CSVs.
        (HERE / "execution_location.json").write_text(json.dumps({"scratch": str(scratch)}), encoding="utf-8")
        try:
            client.execute()
        finally:
            # The supplied manager is externally owned, so nbclient does not
            # shut it down. Release its cwd before Windows temporary cleanup.
            manager.shutdown_kernel(now=True)
        nbformat.validate(notebook)
        nbformat.write(notebook, path)
        shutil.copytree(scratch / "rollback_smc_results", HERE / "results", dirs_exist_ok=True)
        shutil.copy2(scratch / "rollback_smc_results.zip", HERE / "rollback_smc_results.zip")
    code_cells = [c for c in notebook.cells if c.cell_type == "code"]
    assert all(c.execution_count is not None for c in code_cells)
    assert not any(o.output_type == "error" for c in code_cells for o in c.outputs)
    check = {"status": "success", "code_cells_executed": len(code_cells), "all_code_cells_executed": True,
             "empty_working_directory": True, "errors": 0, "seconds": time.perf_counter()-start}
    (HERE / "execution_verification.json").write_text(json.dumps(check, indent=2), encoding="utf-8")
    print(json.dumps(check, indent=2), flush=True)


if __name__ == "__main__":
    main()
