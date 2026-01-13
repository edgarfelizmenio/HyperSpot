from pathlib import Path

from lightning.pytorch.callbacks import Callback

# get the experiment run 
def get_experiment_run(project_dir, task, version: str | int = "latest"):
    run_dir = Path(project_dir) / task
    
    latest_dir = None
    for d in run_dir.iterdir():
        if d.is_dir() and d.name.startswith("version_"):
            dir_ver = d.name.split("_")[1]
            
            if dir_ver.isdigit():
                dir_ver = int(dir_ver)
                if latest_dir is None or latest_dir < dir_ver:
                    latest_dir = dir_ver
                if isinstance(version, int) and version == dir_ver:
                    return d

    if version == "latest":
        return d

    return None

class TrainerStatusLogCallback(Callback):
    def __init__(self, logger):
        super(TrainerStatusLogCallback, self).__init__()
        self.logger = logger
    
    def on_train_start(self, trainer, pl_module):
        self.logger.info(f"PROCESS {trainer.global_rank}: Training...")
        print("training")
    
    def on_train_end(self, trainer, pl_module):
        self.logger.info(f"PROCESS {trainer.global_rank}: Training finished.")

    def on_predict_start(self, trainer, pl_module):
        self.logger.info(f"PROCESS {trainer.global_rank}: Predicting...")

    def on_predict_end(self, trainer, pl_module):
        self.logger.info(f"PROCESS {trainer.global_rank}: Prediction finished.")