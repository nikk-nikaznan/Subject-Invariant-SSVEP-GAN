# Subject-Invariant-SSVEP-GAN
Code to accompany our International Conference on Pattern Recognition (ICPR) paper entitled -
[Leveraging Synthetic Subject Invariant EEG Signalsfor Zero Calibration BCI](https://arxiv.org/pdf/2007.11544.pdf).

The code is structured as follows:

- `src/persubject ` contains codes for SSVEP Classification for a Single Subject; 
- `src/leaveoneout ` contains codes for SSVEP Classification for Unseen Subject : Leave-One-Out;
- `src/crosstask ` contains codes for SSVEP Classification for Unseen Subject : Cross-Task;
- `CNN_Subject_Classification.py ` contains code for subject-biometric classification network;
- `CNN_Subject_softmax.py  ` contains code for Softmax probability values taken for the generated data;
- `DC-GAN.py  ` Our DC-GAN based model for generating SSVEP-based EEG data;
- `AC-GAN.py  ` Our AC-GAN based model for generating SSVEP-based EEG data;
- `SIS-GAN.py  ` Our proposed SIS-GAN based model for generating subject invariant SSVEP-based EEG data;
- `CNN_pretrainsubject.py ` contains code for pre-training subject-biometric classification network;
- `CNN_SSVEP_Classification.py ` our SSVEP classification network;

## Dependencies and Requirements
The code has been designed to support python 3.6+ only. The project has the following dependencies and version requirements:

- torch=1.1.0+
- numpy=1.16++
- python=3.6.5+
- scipy=1.1.0+

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
