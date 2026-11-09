"""Small end-to-end structural test without the expensive final replications."""
import tempfile
from pathlib import Path
import pandas as pd
import deep_study as s

rows = []
for rep in range(3):
    run_rows, _ = s.run_deep(s.make_seed(10, rep))
    for row in run_rows:
        row["run"] = rep
    rows.extend(run_rows)
frame = pd.DataFrame(rows)
summary = s.summarize_estimates(frame)
diagnostics = s.diagnostic_quantiles(frame)
assert frame.shape[0] == 3*20*2*2
assert summary.shape[0] == 20*2*2*2

true_latency = s.median_crude_latency
s.median_crude_latency = lambda count, stream, level, reps=8: .0005+count*1e-7
budgets = s.prepare_crude_budgets(frame)
s.median_crude_latency = true_latency
assert budgets.shape[0] == 20*2*2
assert len(s.crude_prefix_estimates([64, 128, 64], s.make_seed(60, 1))) == 3
budgets["crude_n"] = 256
s.CRUDE_REPS = 2
crude = s.run_crude_budgets(budgets)
crude_summary = s.summarize_crude(crude, budgets)
target = Path(tempfile.gettempdir())
s.plot_single_run(frame[frame.run == 0], target)
s.plot_diagnostics(frame, target)
s.plot_replications(summary, crude_summary, target)
print({"frame": frame.shape, "summary": summary.shape,
       "diagnostics": diagnostics.shape, "budgets": budgets.shape,
       "crude": crude.shape, "crude_summary": crude_summary.shape})
