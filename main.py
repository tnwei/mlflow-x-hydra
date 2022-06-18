"""
Example multirun command: python main.py -m runname=testrun lr=1e-4,5e-5,1e-5,5e-6
"""

# Import everything we need
# -------------------------
import sys
from pathlib import Path
import random
import matplotlib.pyplot as plt

# Import hydra and mlflow
# -----------------------
import hydra
import mlflow

# Obtain current working directory outside Hydra
# ----------------------------------------------
# Why: wd is changed temporarily for each Hydra run
# Original wd can still be retrieved at runtime with
# hydra.utils.get_original_cwd()
# But I prefer sticking to std lib when possible
wd = Path(__file__).parent.resolve()

# Define MLflow tracking and artifact storage URI
# -----------------------------------------------
# This is required whenever a new MLflow experiment is defined
# Do not change halfway through! Stick to one
# Migrating mlflow backend is non-trivial

# Default config: local backend + local artifact store
# Command for UI: mlflow ui
# Format for folder is `file://` + absolute file path, following file URI scheme
TRACKING_URI = f"file://{wd}/mlruns"  # This is default location
ARTIFACT_URI = TRACKING_URI

# Alternative config: sqlite backend + local artifact store
# Command for UI: mlflow ui --backend-store-uri sqlite:///mlruns.sqlite
# Format for sqlite is `sqlite:///` + absolute file path. Three slashes!
# TRACKING_URI = f"sqlite:///{wd}/mlruns.sqlite"
# ARTIFACT_URI = f"file://{wd}/mlruns"

# More options available but omitted for brevity
# See https://www.mlflow.org/docs/latest/tracking.html#how-runs-and-artifacts-are-recorded

mlflow.set_tracking_uri(TRACKING_URI)

# Main logic
# ----------
# Main training logic needs to be:
# (1) packaged into a function
# (2) wrapped with hydra.main() decorator
# (3) and called at the end of the script
# for Hydra to work
#
# Config path is dir to where the config is stored
# Config name is the name of the config file
@hydra.main(config_path="conf", config_name="train.yaml")
def main(cfg):
    # Hydra config parsing
    # --------------------
    # cfg is literally a dict of everything defined in the conf file
    # or passed from CLI
    # refer to conf.yaml for example
    print(cfg)

    # Unpack variables that will be used multiple times throughout the script
    # -----------------------------------------------------------------------
    # Why: This is to keep the script modular if migrating away from Hydra
    # Saves time if want to switch to some other runner e.g. Weights and Biases sweeps
    n_epochs = cfg.n_epochs
    lr = cfg.lr
    batch_size = cfg.batch_size

    # Set MLFlow experiment
    # ---------------------
    # All runs recorded under the same exp name will show up on the same table in MLFlow
    # Create exp if not exists
    if mlflow.get_experiment_by_name(cfg.expname) is None:
        mlflow.create_experiment(
            name=cfg.expname, artifact_location=ARTIFACT_URI + "/" + cfg.expname
        )
    mlflow.set_experiment(cfg.expname)

    # Explicitly set new MLFlow run
    # -----------------------------
    # Explicitly set a new run, else running w/ hydra will pick up from a prev run
    # Why 1: Although any mlflow logging will create it automatically,
    # you won't have access to the run metadata
    # Why 2: Allows setting run_name, else would be blank
    # Left blank as default in conf.yaml, but can be specified in CLI
    # Useful when doing multiruns under the same exp
    # Note: Not setting run_id, since the randomly generated UUID is convenient
    activerun = mlflow.start_run(run_name=cfg.runname)

    # Explicitly configure savedir
    # ----------------------------
    # Why: Create consistent savedir for all MLflow exps instead of using Hydra
    # Format: outputs/expname/runid/
    # Hydra's approach will have one savedir for each run
    # Use run ID instead of run name as latter isn't always defined
    savedir = wd / "outputs" / f"{cfg.expname}/{activerun.info.run_id}"
    if not savedir.exists():
        savedir.mkdir(parents=True)

    # Log hyperparameters
    # -------------------
    # Params can only be logged once, while metrics are logged over time

    # Log dict of parameters
    mlflow.log_params(cfg)

    # Log one parameter
    # Note: found logging the command that triggered the file to run to be useful
    mlflow.log_param("argv", " ".join(sys.argv))

    # Data loading and preprocessing
    # ------------------------------
    # ...
    # YOUR CODE HERE
    # ...

    # Main training loop
    # ------------------

    for epoch in range(n_epochs):
        # Model training
        # --------------
        # ...
        # YOUR CODE HERE
        # ...

        # Compute evaluation metrics
        # --------------------------
        print(f"Calculating metrics at step {epoch}/{n_epochs}")
        eval_metrics = {"acc": epoch / n_epochs}

        # Log metrics
        # -----------
        # Log dict of metrics
        mlflow.log_metrics(eval_metrics, step=epoch)

        # Log one metric
        mlflow.log_metric("luck", random.random(), step=epoch)

        # Save training outputs
        # ---------------------
        # Store artifacts by step in the defined savedir
        (savedir / f"{i:02d}.csv").touch()

        # Create a plot
        fig, ax = plt.subplots()
        ax.plot([0, 1], [2, 3])

        # If needed, log training outputs to MLflow artifact store
        # --------------------------------------------------------
        # Log one file
        mlflow.log_artifact(local_path=savedir / f"{i:02d}.csv")

        # Log the whole dir
        mlflow.log_artifacts(local_dir=savedir)

        # Log a figure
        mlflow.log_figure(fig, artifact_file=f"{i:02d}.png")

    # End MLflow run
    # --------------
    # Alternative: wrap code in `with mlflow.start_run() as run:`
    mlflow.end_run()


if __name__ == "__main__":
    main()
