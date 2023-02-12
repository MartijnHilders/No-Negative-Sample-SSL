import wandb
from utils.experiment_utils import load_yaml_to_dict

config = load_yaml_to_dict("configs/sweeps/sweep_vicreg_inertial_skeleton_mmact.yaml")
sweep_id = wandb.sweep(config)
wandb.agent(sweep_id)
