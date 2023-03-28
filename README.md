# Machine learning for QC phase identification

Supporting Information for the paper "Deep learning enables rapid identification of a previously unknown quasicrystal solely from multiphase powder diffraction patterns"

## Repository
In this repository, we provide an implementation of CNN models with an example trained on a synthetic dataset.

1. Compile cython code (generator.pyx): python setup.py build_ext --inplace
2. Run the tuning.py to tuning the model
3. Run the training.py to train the models
4. Run the screening.py to screen your powder x-ray diffraction patterns

### Requirements
- python >= 3.8
- cython >= 0.29
- numpy >= 1.24
- scipy >= 1.10
- tensorflow >= 2.11
- keras >= 2.11
- optuna >= 3.1
