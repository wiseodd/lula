# LULA --- Learnable Uncertainty under Laplace Approximations

Companion code for the paper "Learnable Uncertainty under Laplace Approximations" (UAI 2021).

* ArXiv version: <https://arxiv.org/abs/2010.02720>.
* Citation:
```
@inproceedings{kristiadi2021learnable,
  title={Learnable uncertainty under {L}aplace approximations},
  author={Kristiadi, Agustinus and Hein, Matthias and Hennig, Philipp},
  booktitle={UAI},
  year={2021},
}
```

## LULA Implementation

The source code for LULA is in the `lula` directory
* `lula/model` contains classes for augmenting a MAP-trained network with LULA units
* `lula/train` contains methods for specifically training the associated parameters of LULA units
* `lula/util` contains useful utilities, both for the construction and training of LULA


## Reproducing the Paper

To reproduce all experimental results in the paper, first run:
```
./train.sh
./repeat.sh
```
The model files will be stored in `pretrained_models` directory, while the raw experimental results will be in `results` (the results used in the paper are already in there).

Then, to obtain the plots/tables used in the paper, run
```
python plot_MNISTC.py  # rotated-MNIST
python plot_CIFAR10C.py
python table_calib.py  # calibration, i.e. accuracy and ECE
python table_OOD.py --metrics mmc_fpr95
```

## License

```
MIT License

Copyright (c) 2021 Agustinus Kristiadi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
