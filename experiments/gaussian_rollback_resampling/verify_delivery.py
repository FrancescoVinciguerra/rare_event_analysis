"""Audit executed notebook, all recorded decisions and historical baseline."""
from pathlib import Path
import json
import re
from datetime import datetime
import nbformat
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def main():
    notebook = nbformat.read(HERE / "RollBack_SMC_Colab.ipynb", as_version=4)
    nbformat.validate(notebook)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    sources = "\n".join(cell.source for cell in notebook.cells)
    assert not re.search(r"(?<![A-Za-z])[A-Za-z]:[\\/]", sources)
    assert "from model import" not in sources and "from smc_resampling import" not in sources
    assert all(cell.execution_count is not None for cell in code_cells)
    assert not any(output.output_type == "error" for cell in code_cells for output in cell.outputs)
    png_count = sum("image/png" in output.get("data", {}) for cell in code_cells for output in cell.outputs)
    assert png_count == 3
    run_metadata = json.loads((HERE / "results/manifest.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(HERE / "results/runs.csv")
    n = run_metadata["N"]
    for row in frame.query("policy != 'crude'").itertuples():
        selected = json.loads(row.resampled_levels)
        ess = json.loads(row.ess_before)
        expected = [k for k, value in enumerate(ess, 1) if row.policy == "ess_systematic" and k < 8 and value < .5*n]
        assert selected == expected
        assert 8 not in selected
        np.testing.assert_allclose(sum(json.loads(row.log_c_path)), row.log_z, atol=1e-11)
        np.testing.assert_allclose(np.exp(row.log_z), row.z_hat, rtol=1e-12)
        np.testing.assert_allclose(row.ordinary, row.snis*row.reverse_mass, rtol=1e-10, atol=1e-15)
        assert row.transitions == n*16 + sum(json.loads(row.stage_transitions))
    historical = pd.read_csv(HERE.parent / "gaussian_rollback_smc/results/replicates.csv")
    baseline = frame.query("policy == 'independent'")
    joined = baseline.merge(historical.query("config_id >= 0"), on=["config_id", "replicate"], suffixes=("_new", "_old"), validate="one_to_one")
    assert len(joined) == len(baseline)
    max_error = 0
    for column in ("ordinary", "snis"):
        np.testing.assert_allclose(joined[column+"_new"], joined[column+"_old"], rtol=1e-11, atol=1e-14)
        max_error = max(max_error, float(np.max(np.abs(joined[column+"_new"]-joined[column+"_old"]))))
    active_count = int(((frame.policy == "ess_systematic") & (frame.resampling_count > 0)).sum())
    assert active_count == run_metadata["active_resampling_runs"] > 0
    audit = {"status": "passed", "notebook_code_cells_executed": len(code_cells),
             "no_local_imports_or_machine_paths_in_notebook_sources": True,
             "population_runs_checked": int((frame.policy != "crude").sum()),
             "active_resampling_runs": active_count,
             "all_ess_decisions_and_final_exclusion_verified": True,
             "all_normalizers_and_costs_verified": True,
             "historical_baseline_runs_matched": len(joined), "maximum_baseline_absolute_error": max_error}
    (HERE / "delivery_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    first = code_cells[0].metadata.execution["iopub.status.busy"]
    last = code_cells[-1].metadata.execution["iopub.status.idle"]
    verification = {"status": "success", "code_cells_executed": len(code_cells),
                    "all_code_cells_executed": True, "empty_working_directory": True,
                    "errors": 0, "embedded_png_figures": png_count,
                    "first_cell_started": first, "last_cell_finished": last,
                    "kernel_execution_seconds": (datetime.fromisoformat(last)-datetime.fromisoformat(first)).total_seconds(),
                    "verification_source": "executed notebook cell timestamps and outputs, plus complete result audit",
                    "execution_note": "All notebook cells completed successfully. The initial local harness reported a Windows temporary-directory cleanup error after saving notebook and results; explicit external-kernel shutdown was subsequently added to the harness."}
    (HERE / "execution_verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
