# Subject-Invariant-SSVEP-GAN

[![Arxiv](https://img.shields.io/badge/ArXiv-2112.06567-orange.svg)](https://arxiv.org/abs/2007.11544)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/pykeen)](https://img.shields.io/pypi/pyversions/pykeen)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Code to accompany our International Conference on Pattern Recognition (ICPR) paper entitled -
[Leveraging Synthetic Subject Invariant EEG Signalsfor Zero Calibration BCI](https://arxiv.org/pdf/2007.11544.pdf).

The code is structured as follows:

- `cnn_subject_classification.py` contains code for subject-biometric classification network;
- `cnn_subject_softmax.py` contains code for Softmax probability values taken for the generated data;
- `sis_gan.py` Our proposed SIS-GAN based model for generating subject invariant SSVEP-based EEG data;
- `cnn_pretrainsubject.py` contains code for pre-training subject-biometric classification network;
- `cnn_ssvep_Classification.py` our SSVEP classification network;
- `models.py` contains all the related models;

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

- The `sample_data` folder contains randomly generated data that is used to represent the shape of the input data. It is important to note this is not the real EEG data.

- Create the pre-train subject weight. This can be done by running:

```bash
uv run python -m sis_gan.cnn_pretrainsubject
```

- Then, train sis_gan model by using the pretrain subject weight as a frozen network.

```bash
uv run python -m sis_gan.generate_sisgan
```

- Lastly, evaluate the performance of the generated synthetic data by running:

```bash
uv run python -m sis_gan.cnn_ssvep_classification
```

Model configurations are controlled by using yaml files that can be found in the config directory. This can be changed to customise the model accordingly.

## Cite

Please cite the associated paper for this work if you use this code:

```
@inproceedings{aznan2021leveraging,
  title={Leveraging Synthetic Subject Invariant EEG Signals for Zero Calibration BCI},
  author={Aznan, Nik Khadijah Nik and Atapour-Abarghouei, Amir and Bonner, Stephen and Connolly, Jason D and Breckon, Toby P},
  booktitle={2020 25th International Conference on Pattern Recognition (ICPR)},
  pages={10418--10425},
  year={2021},
  organization={IEEE}
}
```
