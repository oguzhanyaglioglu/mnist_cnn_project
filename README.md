# MNIST CNN Project

This project is a clean and step-by-step PyTorch CNN implementation for the MNIST handwritten digit dataset.

The goal of the project is not only to train a working model, but also to understand each part of the pipeline clearly:
- configuration
- data loading
- transforms
- model architecture
- training loop
- evaluation
- checkpoint saving/loading
- confusion matrix
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
    ├── data/
    └── best_mnist_cnn.pt
```

## Features

- CNN model for MNIST classification
- Training and evaluation pipeline
- Checkpoint saving and loading
- Training history tracking
- Loss and accuracy plots
- Confusion matrix visualization
- Misclassified image visualization
- Separate debug modules for:
  - data
  - model
  - training

## Requirements

- Python 3.10+ recommended
- PyTorch
- torchvision
- matplotlib

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

## How to Run

Go into the `src` folder and run:

```bash
python main.py
```

## Main Modes

In `main.py`, there are two main flows:

### Normal project flow
```python
run_project(cfg)
# run_debug(cfg)
```

This runs:
- training
- history printing
- loss/accuracy plots
- one-batch prediction
- misclassified image visualization
- confusion matrix

### Debug flow
```python
# run_project(cfg)
run_debug(cfg)
```

This runs the debug functions for:
- config
- seed
- transforms
- dataset split
- dataloaders
- model shape checks
- one-batch loss
- one training step

Some debug functions that require a saved checkpoint may stay commented out until a model checkpoint is available.

## Model

The main model used in this project is `SimpleCNN2`.

Architecture summary:
- Conv2d(1 → 8)
- ReLU
- MaxPool2d
- Conv2d(8 → 16)
- ReLU
- MaxPool2d
- Flatten
- Linear(16 * 7 * 7 → 10)

## Training Output

During training, the project tracks:
- `train_loss`
- `train_acc`
- `test_loss`
- `test_acc`

These values are stored in a history dictionary and then visualized with plots.

## Evaluation Tools

The project includes:
- `predict_one_batch(...)`
- `plot_history(...)`
- `show_confusion_matrix(...)`
- `show_misclassified_images(...)`

These help inspect model behavior beyond just accuracy numbers.

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

## Notes

- The checkpoint path is controlled by `Config`.
- The project is designed to be educational and modular.
- Debug functions are separated from the main workflow to keep the main pipeline clean.

## Future Improvements

Possible next steps:
- split entry points into `train.py` and `debug_main.py`
- add README examples with output screenshots
- add per-class accuracy
- add precision/recall/F1 metrics
- make checkpoint/output folders more structured
