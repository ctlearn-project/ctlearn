"""Helpers for testing CWL workflows."""

import json
import os
import shutil
import subprocess
import sys
from importlib.resources import as_file
from pathlib import Path

from ctapipe.core.provenance import json_config_handler


def run_cwl_workflow(tmp_path, workflow, job):
    """Run a CWL workflow and return its output directory and parsed outputs.

    The workflow may be a filesystem path or a package resource. The job is
    serialized to JSON in ``tmp_path``, where the output directory is also created.
    """
    cwltool = shutil.which("cwltool")
    assert cwltool is not None, "cwltool is required for workflow tests"

    job_path = tmp_path / "cwltool_job.json"
    with job_path.open("w") as file:
        json.dump(job, file, default=json_config_handler, indent=4)
    outdir = tmp_path / "results"
    env = os.environ.copy()
    env["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{env['PATH']}"
    with as_file(workflow) as workflow_path:
        result = subprocess.run(
            [
                cwltool,
                "--no-container",
                "--outdir",
                str(outdir),
                str(workflow_path),
                str(job_path),
            ],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
    assert result.returncode == 0, result.stderr
    return outdir, json.loads(result.stdout)
