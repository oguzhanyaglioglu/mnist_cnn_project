from config import Config
from data_utils import build_dataloaders
from train_utils import run_training
from utils import set_seed, load_json, save_json
import argparse

from eval_utils import (
    plot_history,
    plot_lr_curve,
    predict_one_batch,
    build_confusion_matrix,
    show_confusion_matrix,
    show_misclassified_images,
    get_top_confusions,
    save_final_summary_json
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
    print(f"lr={cfg.lr} | batch_size={cfg.batch_size} | epochs={cfg.epochs} | hidden_dim={cfg.hidden_dim} | dropout={cfg.dropout_rate} "
          f"wd={cfg.weight_decay} | scheduler_name={cfg.scheduler_name}")

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
        "hidden_dim": cfg.hidden_dim,
        "dropout_rate": cfg.dropout_rate,
        "weight_decay": cfg.weight_decay,
        "scheduler_name": cfg.scheduler_name,
        "scheduler_config": None,
        "best_test_acc": max(history["test_acc"]),
        "best_test_loss": min(history["test_loss"]),
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
        Config(
            run_name="exp_01_baseline_lr1e3_bs64",
            lr=1e-3,
            batch_size=64,
            epochs=10,
            scheduler_name=None
        ),
        Config(
            run_name="exp_02_lr5e4_bs64",
            lr=5e-4, batch_size=64,
            epochs=10,
            scheduler_name=None
        ),
        Config(
            run_name="exp_03_lr1e3_bs128",
            lr=1e-3,
            batch_size=128,
            epochs=10,
            scheduler_name=None
        ),
        Config(
            run_name="exp_04_steplr_gamma01_lr1e3_bs64",
            lr=1e-3,
            batch_size=64,
            epochs=10,
            scheduler_name="step",
            step_size=3,
            gamma=0.1
        ),
        Config(
            run_name="exp_05_gamma05_step5_steplr_lr1e3_bs64",
            lr=1e-3,
            batch_size=64,
            epochs=10,
            scheduler_name="step",
            step_size=5,
            gamma=0.5
        ),
        Config(
            run_name="exp_06_plateau_factor05_pat1_lr1e3_bs64",
            lr=1e-3,
            batch_size=64,
            epochs=10,
            scheduler_name="plateau",
            plateau_factor=0.5,
            plateau_patience=1
        ),
        Config(
            run_name="exp_07_pat0_plateau_factor05_ep12_lr1e3_bs64",
            lr=1e-3,
            batch_size=64,
            epochs=12,
            scheduler_name="plateau",
            plateau_factor=0.5,
            plateau_patience=0
        ),
        Config(
            run_name="exp_08_wd1e4_plateau_pat0_ep12_lr1e3_bs64",
            lr=1e-3,
            batch_size=64,
            epochs=12,
            weight_decay=1e-4,
            scheduler_name="plateau",
            plateau_factor=0.5,
            plateau_patience=0
        ),
        Config(
            run_name="exp_09_wd1e5_plateau_pat0_ep12_lr1e3_bs64",
            lr=1e-3,
            batch_size=64,
            epochs=12,
            weight_decay=1e-5,
            scheduler_name="plateau",
            plateau_factor=0.5,
            plateau_patience=0
        ),

        Config(
            run_name="exp_10_dense128_plateau_pat0_ep12_lr1e3_bs64",
            lr=1e-3,
            batch_size=64,
            epochs=12,
            weight_decay=0.0,
            dropout_rate=0.0,
            hidden_dim=128,
            scheduler_name="plateau",
            plateau_factor=0.5,
            plateau_patience=0
        ),
        Config(
            run_name="exp_11_drop03_dense128_plateau_pat0_ep12_lr1e3_bs64",
            lr=1e-3,
            batch_size=64,
            epochs=12,
            weight_decay=0.0,
            dropout_rate=0.3,
            hidden_dim=128,
            scheduler_name="plateau",
            plateau_factor=0.5,
            plateau_patience=0
        ),

        Config(
            run_name="exp_12_drop01_dense128_plateau_pat0_ep12_lr1e3_bs64",
            lr=1e-3,
            batch_size=64,
            epochs=12,
            weight_decay=0.0,
            dropout_rate=0.1,
            hidden_dim=128,
            scheduler_name="plateau",
            plateau_factor=0.5,
            plateau_patience=0
        ),

        Config(
            run_name="exp_13_wd1e5_drop01_dense128_plateau_pat0_ep12_lr1e3_bs64",
            lr=1e-3,
            batch_size=64,
            epochs=12,
            weight_decay=1e-5,
            dropout_rate=0.1,
            hidden_dim=128,
            scheduler_name="plateau",
            plateau_factor=0.5,
            plateau_patience=0
        )

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
            f"wd={r['weight_decay']} | "
            f"hidden_dim={r['hidden_dim']} | "
            f"dropout={r['dropout_rate']} | "
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

    scheduler_config = best_experiment.get("scheduler_config") or {} # {} -> schedulrr name verilip config ayarları verilmediği durumlara önlem

    # Daha önceki ayarlarım, ilk üç config ayarındaki gibi sabit ve keskindi.
    # Fakat bu daha önceki deneylerde var olmayan weight decay, dropout vb optimizasyon ayarları için hata riski taşıyordu.
    # Bunun için bütün configlerde var olan ayarlar haricindekiler için get kullanımına geçildi.
    # Fonksiyonun kullanımı şu şekilde -> get(a, b) = eğer fonksiyonda verilen ayar varsa onu (a'yı) kullan, yoksa b'yi kullan
    # B ayarları için refarans olarak config dosyasındaki ayarlar alındı, bu default'a güvenli bir dönüş sağladı.

    cfg = Config(
        run_name=best_experiment["run_name"],
        lr=best_experiment["lr"],
        batch_size=best_experiment["batch_size"],
        hidden_dim=best_experiment.get("hidden_dim", 0),
        dropout_rate=best_experiment.get("dropout_rate", 0.0),
        weight_decay=best_experiment.get("weight_decay", 0.0),
        scheduler_name=best_experiment.get("scheduler_name", None),
    )

    if cfg.scheduler_name == "plateau":
        cfg.plateau_factor = scheduler_config.get("factor", 0.1)
        cfg.plateau_patience = scheduler_config.get("plateau_patience", 2)

    elif cfg.scheduler_name == "steplr":
        cfg.step_size = scheduler_config.get("step_size", 3)
        cfg.gamma = scheduler_config.get("gamma", 0.1)

    print("\n[best_experiment_loaded]")
    print(best_experiment)

    return cfg

def save_final_summary(best_experiment: dict, cm, save_path: str) -> None:

    top_confusions = get_top_confusions(cm, top_k=5)

    lines = [
        "FINAL MODEL SUMMARY",
        "=" * 60,
        f"run_name: {best_experiment['run_name']}",
        f"best_test_acc: {best_experiment['best_test_acc']:.4f}",
        f"best_test_loss: {best_experiment['best_test_loss']:.4f}",
        "",
        "CONFIG",
        "-" * 60,
        f"lr: {best_experiment['lr']}",
        f"batch_size: {best_experiment['batch_size']}",
        f"hidden_dim: {best_experiment.get('hidden_dim')}",
        f"dropout_rate: {best_experiment.get('dropout_rate')}",
        f"weight_decay: {best_experiment.get('weight_decay')}",
        f"scheduler_name: {best_experiment.get('scheduler_name')}",
        f"scheduler_config: {best_experiment.get('scheduler_config')}",
        "",
        "TOP CONFUSIONS",
        "-" * 60,
    ]

    for true_label, pred_label, count in top_confusions:
        lines.append(f"true={true_label} -> pred={pred_label} | count={count}")

    lines += [
        "",
        "SHORT COMMENT",
        "-" * 60,
        "Best model uses hidden_dim=128, dropout=0.1, a plateau scheduler, and no weight decay.",
        "Most errors occur between visually similar handwritten digits."
    ]

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def evaluate_best_run() -> None:
    cfg = load_best_experiment_config()
    summary_cfg = Config()

    print("\n[evaluating best run]")
    print(f"run_name: {cfg.run_name}")
    print(f"ckpt_path: {cfg.ckpt_path}")
    print(f"history_path: {cfg.history_path}")

    run_from_saved_history(cfg)

    predict_one_batch(cfg)
    cm = build_confusion_matrix(cfg)
    show_confusion_matrix(cfg, cfg.outputs_dir)
    show_misclassified_images(cfg, cfg.outputs_dir)

    best_experiment = load_json(summary_cfg.best_experiment_path)
    save_final_summary(best_experiment, cm, summary_cfg.final_summary_path)
    save_final_summary_json(best_experiment, cm, summary_cfg.final_summary_path_json)

    print(f"\n[saved txt] {summary_cfg.final_summary_path}")
    print(f"\n[saved json] {summary_cfg.final_summary_path_json}")


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

def get_cli_args():

    parser = argparse.ArgumentParser(description="MNIST CNN project runner") # parser nesnesi oluşturuyorum
    parser.add_argument( # terminalden hangi bilgileri almak istediğimi tanımlıyorum
        "--mode",
        type=str,
        default="eval",
        choices=["train", "eval", "full", "debug"],
        help="Choose which pipeline mode to run."
    )

    return parser.parse_args() # kullanıcının verdiği komutu okuyup, komuttaki bilgileri (mode vs) ayırıyorum

if __name__ == "__main__":
    # cfg = Config()
    # print(cfg)  -> <__main__.Config object at 0x000001F3A8C2D7F0> -> normal class çıktısı
    # print(cfg) -> Config(seed=42, batch_size=64, lr=0.001) -> dataclass çıktısı

    # run_mode = "eval"
    args = get_cli_args()
    print(args)
    run_by_mode(args.mode)

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



















