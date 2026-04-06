from config import Config
from data_utils import build_dataloaders
from train_utils import run_training
from utils import set_seed

from eval_utils import (
    plot_history,
    predict_one_batch,
    show_confusion_matrix,
    show_misclassified_images,
)

from debug_data import (
    debug_transforms,
    debug_mninst_batch_stats,
    debug_shuffle_effect,
    debug_train_test_split,
    debug_dataloaders_one_batch,
)

from debug_model import (
    debug_model_forward_shape,
    debug_model_pool_shape,
    debug_model_classifier_shape,
)

from debug_train import (
    debug_config,
    debug_seed,
    debug_loss_one_batch,
    debug_train_one_step,
    debug_load_best_and_predict,
    debug_misclassified,
)

def run_project(cfg: Config) -> None:
    train_loader, test_loader = build_dataloaders(cfg)
    history = run_training(cfg, train_loader, test_loader)
    #predict_one_batch(cfg)
    print("\n[history]")
    print(history)
    #plot_history(history, cfg.outputs_dir)
    #show_misclassified_images(cfg, cfg.outputs_dir)
    #show_confusion_matrix(cfg, cfg.outputs_dir)

def run_debug(cfg: Config) -> None:
    debug_config(cfg)
    debug_seed(cfg)

    debug_transforms()
    debug_mninst_batch_stats(cfg)
    debug_shuffle_effect(cfg)
    debug_train_test_split(cfg)
    debug_dataloaders_one_batch(cfg)

    debug_model_forward_shape(cfg)
    debug_model_pool_shape(cfg)
    debug_model_classifier_shape(cfg)

    debug_loss_one_batch(cfg)
    debug_train_one_step(cfg)
    # check point varsa aç
    # debug_load_best_and_predict(cfg)
    # debug_misclassified(cfg, max_show=15)

if __name__ == "__main__":

    cfg = Config()
    set_seed(cfg.seed)

    run_project(cfg)
    # run_debug(cfg)



















