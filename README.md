# Machine Learning for powder diffraction of quasicrystals

Supporting Information for the paper "Deep learning enables rapid identification of a previously unknown quasicrystal solely from multiphase powder diffraction patterns"

## Repository
In this repository, we provide an implementation of a MLqcdiff model.

1. Compile cython code (generator.pyx): `python setup.py build_ext --inplace`
2. Run the training.py to train the models
3. Run the screening.py to screen dataset of powder x-ray diffraction patterns
