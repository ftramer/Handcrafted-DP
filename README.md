# Handcrafted-DP

**This repository contains code to train differentially private models 
with handcrafted vision features.**

These models are introduced and analyzed in:

*Differentially Private Learning Needs Better Features (Or Much More Data)*</br>
**Florian Tramèr and Dan Boneh**</br>
[arXiv:2011.11660](http://arxiv.org/abs/2011.11660)

## Installation

The main dependencies are [pytorch](https://github.com/pytorch/pytorch), 
[kymatio](https://github.com/kymatio/kymatio) 
and [opacus](https://github.com/pytorch/opacus).

You can install all requirements with:
```bash
pip install -r requirements.txt
```

The code was tested with `python 3.7`, `torch 1.6` and `CUDA 10.1`.


## Example Usage and Results

This table presents the main results from our paper. For each dataset, we target a privacy budget of `(epsilon=3, delta=1e-5)`.
We compare three types of models: 
1) Regular CNNs trained "end-to-end" from image pixels.
2) Linear models fine-tuned on top of "handcrafted"
[ScatterNet](https://arxiv.org/abs/1412.8659) features.
3) Small CNNs fine-tuned on ScatterNet features. 

| Dataset  | End-to-end CNN | ScatterNet + linear | ScatterNet + CNN |
| ------------- | ------------- | ------------- | ------------- |
| MNIST  | 98.1%  | **98.7%** | **98.7%**
| Fashion-MNIST  | 86.0%  | **89.7%** | 89.0%
| CIFAR-10  | 59.2%  | 67.0% | **69.3%**


### Determining the Noise Multiplier

The [DP-SGD](https://arxiv.org/abs/1607.00133) algorithm adds noise 
to every gradient update to preserve privacy.
The "noise multiplier" is a parameter that determines the amount of noise 
that is added. 
The higher the noise multiplier, the stronger the privacy guarantees, 
but the harder it is to train accurate models.

In our paper, we compute the noise multiplier so that our fixed privacy budget
of `(epsilon=3, delta=1e-5)` is consumed after some fixed number of epochs.
The noise multiplier can be computed as:
```python
from dp_utils import get_noise_mul
num_samples = 50000
batch_size = 512
target_epsilon = 3
target_delta = 1e-5
epochs = 40
noise_mul = get_noise_mul(num_samples, batch_size, target_epsilon, epochs, target_delta=target_delta)
```

### End-to-end CNNs

To reproduce the results for end-to-end CNNs with the best hyper-parameters from our paper, run
```bash
python3 cnns.py --dataset=mnist --batch_size=512 --lr=0.5 --noise_multiplier=1.23
python3 cnns.py --dataset=fmnist --batch_size=2048 --lr=4 --noise_multiplier=2.15
python3 cnns.py --dataset=cifar10 --batch_size=1024 --lr=1 --noise_multiplier=1.54
```
The noise multipliers are computed so as to consume the privacy budget in 
respectively `40`, `40` and `30` epochs.

### ScatterNet models

To reproduce the results for linear ScatterNet models, run
```bash
python3 baselines.py --dataset=mnist --batch_size=4096 --lr=8 --input_norm=BN --bn_noise_multiplier=8 --noise_multiplier=3.04
python3 baselines.py --dataset=fmnist --batch_size=8192 --lr=16 --input_norm=GroupNorm --num_groups=27 --noise_multiplier=4.05
python3 baselines.py --dataset=cifar10 --batch_size=8192 --lr=4 --input_norm=BN --bn_noise_multiplier=8 --noise_multiplier=5.67
```
And for CNNs fine-tuned on ScatterNet features, run:
```bash
python3 cnns.py --dataset=mnist --use_scattering --batch_size=1024 --lr=1 --input_norm=BN --bn_noise_multiplier=8 --noise_multiplier=1.35
python3 cnns.py --dataset=fmnist --use_scattering --batch_size=2048 --lr=4 --input_norm=GroupNorm --num_groups=27 --noise_multiplier=2.15
python3 cnns.py --dataset=cifar10 --use_scattering --batch_size=8192 --lr=4 --input_norm=BN --bn_noise_multiplier=8 --noise_multiplier=5.67
```

There are a few additional parameters here:
* The `input_norm` parameter determines how the ScatterNet features are normalized. 
We support Group Normalization (`input_norm=GN`) 
and (frozen) Batch Normalization (`input_norm=BN`).
* When using Group Normalization, the `num_groups` parameter specifies the number
of groups into which to split the features for normalization.
* When using Batch Normalization, we first privately compute the mean and variance
of the features across the entire training set. This requires adding noise to 
these statistics. The `bn_noise_multiplier` specifies the scale of the noise. 

When using Batch Normalization, we *compose* the privacy losses of the 
normalization step and of the DP-SGD algorithm.
Specifically, we first compute the Rényi-DP budget for the normalization step, 
and then compute the `noise_multiplier` of the DP-SGD algorithm so that the total
privacy budget is used after a fixed number of epochs:
```python
from dp_utils import get_renyi_divergence, get_noise_mul
rdp = 2 * get_renyi_divergence(1, bn_noise_multiplier)
noise_mul = get_noise_mul(num_samples, batch_size, target_epsilon, epochs, rdp_init=rdp, target_delta=target_delta)
```

### Measuring the Data Complexity of Private Learning

To understand how expensive it currently is to exceed handcrafted features 
with private end-to-end deep learning, we compare the performance of the 
above models on increasingly large training sets.

To obtain a larger dataset comparable to CIFAR-10, we use `500'000` additional
pseudo-labelled tiny images collected by [Carmon et al.](https://github.com/yaircarmon/semisup-adv)

To re-train the above models for `120` epochs on the full dataset of `550'000` images, use:

```bash
python3 tiny_images.py --batch_size=8192 --lr=16 --delta=9.09e-7 --model=linear --use_scattering --bn_noise_multiplier=8 --epochs=120 --noise_multiplier=1.1
python3 tiny_images.py --batch_size=8192 --lr=16 --delta=9.09e-7 --model=cnn --epochs=120 --noise_multiplier=1.1	
python3 tiny_images.py --batch_size=8192 --lr=16 --delta=9.09e-7 --model=cnn --use_scattering --bn_noise_multiplier=8 --epochs=120 --noise_multiplier=1.1
```

For a privacy budget of `(epsilon=3, delta=1/2N)`, where `N` is the size of the 
training data, we obtain the following improved test accuracies on CIFAR-10:

N| End-to-end CNN | ScatterNet + linear | ScatterNet + CNN |
| ------------- | ------------- | ------------- | ------------- |
| 50K  | 59.2%  | 67.0% | **69.3%**
|550K |  **75.8%** | 70.7% | 74.5% |

### Private Transfer Learning
Our paper also contains some results for private transfer learning to CIFAR-10.
For a privacy budget of `(epsilon=2, delta=1e-5)` we get:

 Source Model  | Transfer Accuracy on CIFAR-10 |
| ------------- | ------------- | 
| ResNeXt-29 (CIFAR-100) | 79.6%  | 
| SIMCLR v2 (unlabelled ImageNet) | 92.4%  | 

These results can be reproduced as follows. 
First, you'll need to download the `resnext-8x64d` model from 
[here](https://github.com/bearpaw/pytorch-classification).

Then, we extract features from the source models:
```bash
python3 -m transfer.extract_cifar100
python3 -m transfer.extract_simclr
```
This will create a `transfer/features` directory unless one already exists.

Finally, we train linear models with DP-SGD on top of these features:
```bash
python3 -m transfer.transfer_cifar --feature_path=transfer/features/cifar100_resnext --batch_size=2048 --lr=8 --noise_multiplier=3.32
python3 -m transfer.transfer_cifar --feature_path=transfer/features/simclr_r50_2x_sk1 --batch_size=1024 --lr=4 --noise_multiplier=2.40
```
