# Machine Learning for Powder Diffraction of Quasicrystals

Supporting Information for the following paper

"Deep Learning Enables Rapid Identification of a New Quasicrystal from Multiphase Powder Diffraction Patterns"

Hirotaka Uryu, Tsunetomo Yamada, Koichi Kitahara, Alok Singh, Yutaka Iwasaki, Kaoru Kimura, Kanta Hiroki, Naoki Miyao, Asuka Ishikawa, Ryuji Tamura, Satoshi Ohhashi, Chang Liu, Ryo Yoshida

Advanced Science , 2304546-1-9 published 4/11/2023

DOI:10.1002/advs.202304546
 
## Repository
In this repository, we provide an implementation of MLqcdiff models.

1. Compile the generator.pyx: `python setup.py build_ext --inplace`
2. Run the training.py to train the models
3. Run the screening.py to screen dataset of powder XRD patterns
