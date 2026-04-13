from config import Config
from data_utils import build_dataloaders
from train_utils import run_training
from utils import set_seed, load_json, save_json

from eval_utils import (
    plot_history,
    plot_lr_curve,
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

def run_project(cfg: Config) -> dict:
    print("\n" + "=" * 60)
    print(f"[run] {cfg.run_name}")
    print(f"lr={cfg.lr} | batch_size={cfg.batch_size} | epochs={cfg.epochs}")

    train_loader, test_loader = build_dataloaders(cfg)
    history = run_training(cfg, train_loader, test_loader)

    plot_history(history, cfg.outputs_dir)
    plot_lr_curve(history, cfg.outputs_dir)

    print("\n[history]")
    print(history)

    result = {
        "run_name": cfg.run_name,
        "lr": cfg.lr,
        "batch_size": cfg.batch_size,
        "epochs_ran": len(history["train_loss"]),
        "best_test_acc": max(history["test_acc"]),
        "best_test_loss": min(history["test_loss"]),
        "scheduler_name": cfg.scheduler_name
    }

    if cfg.scheduler_name == "step":
        result["scheduler_config"] = {
            "step_size": cfg.step_size,
            "gamma": cfg.gamma
        }
    elif cfg.scheduler_name == "plateau":
        result["scheduler_config"] = {
            "factor": cfg.plateau_factor,
            "plateau_patience": cfg.plateau_patience
        }
    else:
        result["scheduler_config"] = None

    print("\n[result]")
    print(result)
    return result



def run_from_saved_history(cfg: Config) -> None:
    loaded_history = load_json(cfg.history_path)

    print("\n[loaded history]")
    print(loaded_history)

    plot_history(loaded_history, cfg.outputs_dir)
    plot_lr_curve(loaded_history, cfg.outputs_dir)

def run_hparam_experiments() -> None:

    # deney_listesi oluştur
    # sonuç_listesi oluştur
    #
    # her deney için:
    #     seed ayarla
    #     eğitimi çalıştır
    #     sonucu sakla
    #
    # sonuçları accuracy'ye göre sırala
    # hepsini ekrana yazdır

    experiments = [
        Config(run_name="exp_01_baseline_lr1e3_bs64", lr=1e-3, batch_size=64, epochs=10, scheduler_name=None),
        Config(run_name="exp_02_lr5e4_bs64", lr=5e-4, batch_size=64, epochs=10, scheduler_name=None),
        Config(run_name="exp_03_lr1e3_bs128", lr=1e-3, batch_size=128, epochs=10, scheduler_name=None),
        Config(run_name="exp_04_steplr_lr1e3_bs64", lr=1e-3, batch_size=64, epochs=10, scheduler_name="step", step_size=3, gamma=0.1),
        Config(run_name="exp_05_steplr_step5_gamma05_lr1e3_bs64", lr=1e-3, batch_size=64, epochs=10, scheduler_name="step", step_size=5, gamma=0.5)
    ]

    results = []
    # exp;
    # {
    #     "run_name": "exp_01_baseline_lr1e3_bs64",
    #     "lr": 1e-3,
    #     "batch_size": 64,
    #     "epochs_ran": 10,
    #     "best_test_acc": 0.9821,
    #     "best_test_loss": 0.0542,
    # }

    for cfg in experiments:
        set_seed(cfg.seed)
        result = run_project(cfg)
        results.append(result)

    results.sort(key=lambda x : x["best_test_acc"], reverse=True)


    print("\n" + "=" * 60)
    print("[experiment_summary]")
    for i, r in enumerate(results, start=1):
        scheduler_info = (
            f"{r['scheduler_name']} {r['scheduler_config']}"
            if r["scheduler_name"] is not None
            else "None"
        )
        print(
            f"{i}. {r['run_name']} | "
            f"best_acc={r['best_test_acc']:.4f} | "
            f"best_loss={r['best_test_loss']:.4f} | "
            f"lr={r['lr']} | bs={r['batch_size']} | "
            f"scheduler={scheduler_info} | "
            f"epochs_ran={r['epochs_ran']}"
        )
        # exp; 1. exp_01_baseline_lr1e3_bs64 | best_acc=0.9821 | best_loss=0.0542 | lr=0.001 | bs=64 | epochs_ran=10

    # [experiment summary]
    # 1.exp_01_baseline_lr1e3_bs64 | best_acc = 0.9891 | ...
    # 2.exp_03_lr1e3_bs128 | best_acc = 0.9884 | ...
    # 3.exp_02_lr5e4_bs64 | best_acc = 0.9872 | ...

    summary_cfg = Config()
    save_json(results, summary_cfg.experiment_results_path)
    save_json(results[0], summary_cfg.best_experiment_path)

    print("\n[save files]")
    print(summary_cfg.experiment_results_path)
    print(summary_cfg.best_experiment_path)

def load_best_experiment_config() -> Config:
    summary_cfg = Config()
    best_experiment = load_json(summary_cfg.best_experiment_path)

    best_run_name = best_experiment["run_name"]

    cfg = Config(
        run_name=best_run_name,
        lr=best_experiment["lr"],
        batch_size=best_experiment["batch_size"]
    )

    print("\n[best_experiment_loaded]")
    print(best_experiment)

    return cfg

def evaluate_best_run() -> None:
    cfg = load_best_experiment_config()

    print("\n[evaluating best run]")
    print(f"run_name: {cfg.run_name}")
    print(f"ckpt_path: {cfg.ckpt_path}")
    print(f"history_path: {cfg.history_path}")

    run_from_saved_history(cfg)
    predict_one_batch(cfg)
    show_confusion_matrix(cfg, cfg.outputs_dir)
    show_misclassified_images(cfg, cfg.outputs_dir)

def run_full_pipeline() -> None:
    print("\n" + "=" * 60)
    print("\n[full pipiline started]")

    run_hparam_experiments()
    evaluate_best_run()

    print("\n" + "=" * 60)
    print("[full pipeline finished]")


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

def run_by_mode(mode: str) -> None:
    print("\n" + "=" * 60)
    print(f"[mode] {mode}")

    if mode == "train":
        run_hparam_experiments()

    elif mode == "eval":
        evaluate_best_run()

    elif mode == "full":
        run_full_pipeline()

    elif mode == "debug":
        cfg = Config()
        set_seed(cfg.seed)
        run_debug(cfg)

    else:
        # ValueError -> değişken var fakat içindeki değer yanlışsa(exp; mode="abc") dönen hata türü
        # raise -> bilerek hata fırlatmak, yani program kendi kendi hata vermiyor, biz diyoruz ki mode yanlış hata ver
        raise ValueError(
            f"Unknown mode: {mode}."
            f"Choose from ['train', 'eval', 'full', 'debug']"
        )



if __name__ == "__main__":
    # cfg = Config()
    # print(cfg)  -> <__main__.Config object at 0x000001F3A8C2D7F0> -> normal class çıktısı
    # print(cfg) -> Config(seed=42, batch_size=64, lr=0.001) -> dataclass çıktısı

    run_mode = "train"
    run_by_mode(run_mode)

    # run_full_pipeline()
    # evaluate_best_run()
    # run_hparam_experiment()

    # Tek bir kaydedilmiş run'ın logunu tekrar çizdirmek için
    # cfg = Config(run_name="exp_01_baseline_lr1e3_bs64")
    # run_saved_from_history(cfg)

    # Tek bir run için tahmin / confusion matrix / missclassified görmek için
    # cfg = Config(run_name="exp_01_baseline_lr1e3_bs64")
    # predict_one_batch(cfg)
    # show_confusion_matrix(cfg, cfg.outputs_dir)
    # show_misclassified_images(cfg, cfg.outputs_dir)

    # Debug için
    # cfg = Config()
    # set_seed(cfg.seed)
    # run_debug(cfg)



















