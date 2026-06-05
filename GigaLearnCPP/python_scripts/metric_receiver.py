import json
import os
import site
import sys

wandb_run = None


# Takes in the python executable path, the three wandb init strings, and optionally the current run ID
# Returns the ID of the run (either newly created or resumed)
def init(py_exec_path, project, group, name, id=None):

    global wandb_run

    # Fix the path of our interpreter so wandb doesn't run RLGym_PPO instead of Python
    # Very strange fix for a very strange problem
    sys.executable = py_exec_path

    try:
        import subprocess

        # Ask the target Python executable for its actual site-packages dirs.
        # This works for macOS/Linux virtualenvs as well as other interpreter layouts,
        # instead of assuming a Windows-style "Lib/site-packages" path.
        output = subprocess.check_output(
            [
                py_exec_path,
                "-c",
                "import json, site; print(json.dumps(site.getsitepackages() + [site.getusersitepackages()]))",
            ],
            text=True,
        )
        site_package_dirs = [p for p in json.loads(output) if p]

        for site_packages_dir in site_package_dirs:
            if site_packages_dir not in sys.path:
                sys.path.append(site_packages_dir)
            site.addsitedir(site_packages_dir)

        import wandb
    except Exception as e:
        raise Exception(f"""
			FAILED to import wandb! Make sure GigaLearnCPP isn't using the wrong Python installation.
			Configured Python executable: {py_exec_path}
			This installation's site packages: {site.getsitepackages()}
			Current sys.path: {sys.path}
			Exception: {repr(e)}""")

    print("Calling wandb.init()...")
    if not (id is None) and len(id) > 0:
        wandb_run = wandb.init(
            project=project, group=group, name=name, id=id, resume="allow"
        )
    else:
        wandb_run = wandb.init(project=project, group=group, name=name)
    return wandb_run.id


def add_metrics(metrics):
    global wandb_run
    wandb_run.log(metrics)
