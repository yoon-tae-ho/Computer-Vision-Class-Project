import json
import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/project3-matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

nb = json.loads(Path("Project #3.ipynb").read_text())
code = "\n\n".join(
    "".join(cell.get("source", []))
    for cell in nb["cells"]
    if cell.get("cell_type") == "code"
)

namespace = {"__name__": "__main__"}
started = time.time()
exec(compile(code, "Project #3.ipynb extracted code", "exec"), namespace)

assert namespace["SMOKE_TEST"] is True, "SMOKE_TEST must remain True."
assert namespace["RUN_FINAL_TEST"] is False, "RUN_FINAL_TEST must remain False."

print("RUNNER: notebook definitions loaded")
print("RUNNER: SMOKE_TEST", namespace["SMOKE_TEST"])
print("RUNNER: RUN_FINAL_TEST", namespace["RUN_FINAL_TEST"])
print("RUNNER: device", namespace["DEVICE"])

checks = namespace["run_static_smoke_verification"]()
print("RUNNER: static checks", json.dumps(checks, sort_keys=True))

namespace["run_step2_sanity_checks"]()
print("RUNNER: step2 sanity checks passed")

smoke_result = namespace["run_step3_smoke_training"](download=True)
print("RUNNER: step3 smoke keys", sorted(smoke_result.keys()))
print(
    "RUNNER: step3 final checkpoint",
    smoke_result.get("stage2", {}).get("best_checkpoint_path")
    or smoke_result.get("stage1", {}).get("best_checkpoint_path"),
)

assert namespace["RUN_FINAL_TEST"] is False, "RUN_FINAL_TEST changed before search."
search_result = namespace["run_scenario_model_search"]("exp", 0.1)
print("RUNNER: search keys", sorted(search_result.keys()))
print("RUNNER: selected final checkpoint", search_result.get("selected_config", {}).get("final_checkpoint_path"))

assert namespace["RUN_FINAL_TEST"] is False, "RUN_FINAL_TEST changed before report artifacts."
artifact_summary = namespace["prepare_report_artifacts"]()
print("RUNNER: artifact summary", json.dumps(artifact_summary, sort_keys=True, default=str))

assert namespace["RUN_FINAL_TEST"] is False, "RUN_FINAL_TEST changed during smoke run."
print("RUNNER: elapsed_sec", round(time.time() - started, 2))
