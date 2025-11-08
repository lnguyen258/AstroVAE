from .callback import CALLBACK_REGISTRY, EarlyStopping
from .dataset import Galaxies_ML_Dataset
from .get_config import (
    get_loss_from_config, 
    get_callbacks_from_config,
    get_optim_from_config, 
    get_scheduler_from_config
)
from .plot_hist import plot_history