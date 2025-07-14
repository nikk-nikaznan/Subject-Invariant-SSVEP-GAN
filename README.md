# Subject-Invariant-SSVEP-GAN

[![Arxiv](https://img.shields.io/badge/ArXiv-2112.06567-orange.svg)](https://arxiv.org/abs/2007.11544)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/pykeen)](https://img.shields.io/pypi/pyversions/pykeen)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

Code to accompany our International Conference on Pattern Recognition (ICPR) paper entitled -
[Leveraging Synthetic Subject Invariant EEG Signals for Zero Calibration BCI](https://arxiv.org/pdf/2007.11544.pdf).

## Code Structure

### Main Components

- `generate_sisgan.py` - Our proposed SIS-GAN model for generating subject-invariant SSVEP-based EEG data
- `train_subject_classifier.py` - Subject classification network for biometric pattern detection
- `ssvep_classification.py` - SSVEP frequency classification network
- `plot_subject_softmax.py` - Subject-invariance evaluation using softmax probability analysis
- `models.py` - Neural network architectures for all models

### Supporting Modules

- `utils.py` - Helper functions for data loading, logging, and model utilities

## Dependencies and Requirements

1. Create a new uv environment with Python 3.12:

```bash
uv venv --python 3.12
```

2. Install the required dependencies defined in pyproject.toml:

```bash
uv pip install -e .
```

## How to Use

Please follow the steps below to run the code to train the various components. Please read the **Configuration** section to understand how to modify the training parameters to suit real-world applications. The number of epochs is set to 1 by default for demonstration purposes, but you will need to increase this for meaningful results.

### Complete Zero-Calibration Pipeline (Leave-One-Out)

This is the main pipeline for zero-calibration BCI evaluation, where we simulate having an unseen subject:

**Step 1: Create pretrained subject classification weights**

```bash
uv run python -m sis_gan.train_subject_classifier --validation_strategy loo
```

_This trains subject classifiers using leave-one-out cross-validation, creating pretrained weights for each "unseen" subject scenario._

**Step 2: Generate synthetic EEG data**

```bash
uv run python -m sis_gan.generate_sisgan
```

_Trains the SIS-GAN using data from "seen" subjects (excluding the test subject) and generates synthetic SSVEP signals._

**Step 3: Evaluate SSVEP classification performance**

```bash
uv run python -m sis_gan.ssvep_classification
```

_Tests how well SSVEP frequency classification works when trained on synthetic data and tested on real unseen subject data._

**Step 4: Verify subject-invariance**

```bash
uv run python -m sis_gan.plot_subject_softmax
```

_Analyzes whether the generated synthetic data contains subject-specific biometric patterns. Low, uniformly distributed probabilities indicate successful subject-invariance._

### Subject Classification Only (Stratified Split)

If you only want to evaluate subject classification performance on the full dataset without running the complete pipeline:

```bash
uv run python -m sis_gan.train_subject_classifier --validation_strategy stratified
```

_Performs standard train/test split (80/20) for subject classification evaluation across the entire dataset._

## Data

The `sample_data` folder contains randomly generated data representing the shape of the input EEG data. **Note: This is not real EEG data and is provided only for demonstration purposes.**

## Configuration

Model configurations are controlled using YAML files in the `config` directory. Modify these files to customize:

- Network architectures
- Training hyperparameters
- Data processing parameters
- Validation strategies

**Note:** By default, all models are only trained for a single epoch for demonstration purposes. Adjust the `num_epochs` parameter in the config files to train for longer - you will need to increase the number of epochs for meaningful results.

## Development

### Run pre-commit hooks

```bash
uv pip install -e ".[dev]"
uv run pre-commit install
uv run pre-commit run --all-files
```

## Key Concepts

- **Subject-Invariant**: Generated data should not contain biometric patterns specific to individual subjects
- **Zero-Calibration**: Using synthetic data to train models for completely unseen subjects without any calibration data
- **Leave-One-Out**: Each subject is treated as "unseen" while others are used for training, simulating real-world deployment scenarios

## Citation

Please cite the associated paper if you use this code:

```bibtex
@inproceedings{aznan2021leveraging,
  title={Leveraging Synthetic Subject Invariant EEG Signals for Zero Calibration BCI},
  author={Aznan, Nik Khadijah Nik and Atapour-Abarghouei, Amir and Bonner, Stephen and Connolly, Jason D and Breckon, Toby P},
  booktitle={2020 25th International Conference on Pattern Recognition (ICPR)},
  pages={10418--10425},
  year={2021},
  organization={IEEE}
}
```
