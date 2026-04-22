# MNIST CNN Project

This project is a clean, step-by-step PyTorch CNN implementation for the MNIST handwritten digit dataset.

The goal is not only to train a strong model, but also to understand the full pipeline clearly:
- configuration
- data loading
- transforms
- model architecture
- training loop
- evaluation
- checkpoint saving/loading
- experiment tracking
- confusion matrix analysis
- misclassified sample analysis
- debug functions

## Project Structure

```text
mnist_cnn_project/
├── .venv/
├── outputs/
├── requirements.txt
└── src/
    ├── config.py
    ├── model.py
    ├── data_utils.py
    ├── train_utils.py
    ├── eval_utils.py
    ├── utils.py
    ├── debug_data.py
    ├── debug_model.py
    ├── debug_train.py
    ├── main.py
    └── data/
```

## Features

- CNN model for MNIST classification
- Training and evaluation pipeline
- Checkpoint saving and loading
- Training history tracking
- Loss, accuracy, and learning rate plots
- Hyperparameter experiment tracking
- Best experiment selection and loading
- Confusion matrix visualization
- Misclassified image visualization
- Final text/JSON summary generation
- Separate debug modules for:
  - data
  - model
  - training

## Requirements

- Python 3.10+ recommended
- PyTorch
- torchvision
- matplotlib
- numpy

## Installation

Create and activate a virtual environment first.

### Windows
```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux
```bash
python -m venv .venv
source .venv/bin/activate
```

Then install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
git clone <repo-url>
cd mnist_cnn_project
python -m venv .venv
```

### Windows
```bash
.venv\Scripts\activate
```

### macOS / Linux
```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
cd src
python main.py
```

Before running, choose the mode in `main.py`:

```python
run_mode = "train"   # or "eval", "full", "debug"
run_by_mode(run_mode)
```

## How to Run

Go into the `src` folder and run:

```bash
python main.py
```

## Main Modes

The project uses a mode-based runner in `main.py`.

```python
run_mode = "full"
run_by_mode(run_mode)
```

Available modes:

### `train`
Runs the experiment set and saves:
- per-run checkpoints
- training history JSON files
- loss, accuracy, and learning rate plots
- experiment summary files
- best experiment file

### `eval`
Loads the best saved experiment automatically and runs:
- history plotting
- one-batch prediction
- confusion matrix
- misclassified sample visualization
- final summary generation

### `full`
Runs the full pipeline:
- hyperparameter experiments
- best experiment selection
- best experiment evaluation

### `debug`
Runs the debug utilities for:
- config
- seed
- transforms
- dataset split
- dataloaders
- model shape checks
- one-batch loss
- one training step

## Model

The final best-performing model uses a CNN classifier with one hidden dense layer and light dropout.

### Final Architecture
- Conv2d(1 → 8)
- ReLU
- MaxPool2d
- Conv2d(8 → 16)
- ReLU
- MaxPool2d
- Flatten
- Linear(16 × 7 × 7 → 128)
- ReLU
- Dropout(0.1)
- Linear(128 → 10)

## Training and Experiment Tracking

During training, the project tracks:
- `train_loss`
- `train_acc`
- `test_loss`
- `test_acc`
- `lr`

The project also saves:
- per-run `training_history.json`
- best checkpoint for each run
- `experiment_results.json`
- `best_experiment.json`

## Evaluation Tools

The project includes:
- `predict_one_batch(...)`
- `plot_history(...)`
- `plot_lr_curve(...)`
- `show_confusion_matrix(...)`
- `show_misclassified_images(...)`
- `save_final_summary(...)`
- `save_final_summary_json(...)`

These tools help inspect model behavior beyond just accuracy numbers.

## Debug Modules

### `debug_data.py`
Contains functions related to:
- transforms
- MNIST batch statistics
- shuffle behavior
- train/test split
- dataloader batch inspection

### `debug_model.py`
Contains functions related to:
- forward output shape
- pooling output shape
- classifier/logit output shape

### `debug_train.py`
Contains functions related to:
- config and seed checks
- one-batch loss
- one-step training behavior
- checkpoint prediction checks
- misclassification inspection

## Final Results

### Best Model
- Run name: `exp_12_drop01_dense128_plateau_pat0_ep12_lr1e3_bs64`
- Best test accuracy: 0.9919
- Best test loss: 0.0272

### Best Configuration
- Learning rate: 0.001
- Batch size: 64
- Hidden dimension: 128
- Dropout rate: 0.1
- Weight decay: 0.0
- Scheduler: ReduceLROnPlateau
- Plateau factor: 0.5
- Plateau patience: 0

### Key Findings
- Adding a hidden dense layer improved the model noticeably.
- A small dropout rate of 0.1 improved generalization.
- A larger dropout rate of 0.3 reduced performance slightly.
- Weight decay did not improve the best dense + dropout configuration.
- ReduceLROnPlateau performed better than the tested StepLR setups.

### Top Confusions
The model mainly confused visually similar handwritten digits, especially:
- 2 and 7
- 7 and 2
- 9 and 4
- 6 and 0
- 4 and 9

### Error Analysis
Most errors occur between visually similar handwritten digits rather than completely unrelated classes.

## Notes

- Output paths are controlled by `Config`.
- The project is designed to be modular and educational.
- Debug functions are separated from the main workflow to keep the pipeline clean.
- Final evaluation artifacts are saved inside the selected best run folder.

## Project Status

Current status: Completed

Implemented:
- baseline training pipeline
- experiment tracking
- scheduler comparison
- dense layer comparison
- dropout comparison
- weight decay comparison
- best model selection
- final evaluation
- confusion matrix analysis
- misclassified sample analysis
- final text and JSON summaries

Best final model:
`exp_12_drop01_dense128_plateau_pat0_ep12_lr1e3_bs64`
