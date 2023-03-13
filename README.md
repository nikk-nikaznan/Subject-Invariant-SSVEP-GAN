# Subject-Invariant-SSVEP-GAN

[![Arxiv](https://img.shields.io/badge/ArXiv-2112.06567-orange.svg)](https://arxiv.org/abs/2007.11544)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/pykeen)](https://img.shields.io/pypi/pyversions/pykeen)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Code to accompany our International Conference on Pattern Recognition (ICPR) paper entitled -
[Leveraging Synthetic Subject Invariant EEG Signalsfor Zero Calibration BCI](https://arxiv.org/pdf/2007.11544.pdf).

The code is structured as follows:

- `CNN_Subject_Classification.py ` contains code for subject-biometric classification network;
- `CNN_Subject_softmax.py  ` contains code for Softmax probability values taken for the generated data;
- `SIS-GAN.py  ` Our proposed SIS-GAN based model for generating subject invariant SSVEP-based EEG data;
- `CNN_pretrainsubject.py ` contains code for pre-training subject-biometric classification network;
- `CNN_SSVEP_Classification.py ` our SSVEP classification network;
- `models.py ` contains all the related models;

## Dependencies and Requirements
The code has been designed to support python 3.7+ only. The dependencies for the project can be installed via pip using the requirements.txt as follows:

```shell
$ pip install -r requirements.txt
```
## How to Use
The ```sample_data``` folder contains some random noise data to represent the data shape of the input data. It is not the real EEG data.

First, create the pretrain subject weight. This can be done by using the ```CNNN_pretrainsubject.py```.

Then, train SIS-GAN in ```SIS-GAN.py```by using the pretrain subject weight as a frozen network.

Lastly, evaluate the performance of the generated synthetic data by using ```CNN_SSVEP_classification```.

## Cite

Please cite the associated papers for this work if you use this code:

```
@article{aznan2020leveraging,
  title={Leveraging Synthetic Subject Invariant EEG Signals for Zero Calibration BCI},
  author={Aznan, Nik Khadijah Nik and Atapour-Abarghouei, Amir and Bonner, Stephen and Connolly, Jason D and Breckon, Toby P},
  journal={arXiv preprint arXiv:2007.11544},
  year={2020}
}
```
