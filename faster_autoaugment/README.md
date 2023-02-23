# Faster AutoAugment

This is the official *re*implementation of FasterAutoAugment ([hataya2020a](https://arxiv.org/abs/1911.06987).)

## Requirements

- `Python>=3.8` # Developed on 3.8. It may work with earlier versions.
- `PyTorch==1.13.1`
- `torchvision==0.14.1`
- `kornia==0.5.0`
- `homura==2020.07` # `pip install -U git+https://github.com/moskomule/homura@v2020.07`
- `hydra-core==1.3.1`
- `omegaconf==2.3.0`

## How to Run

Faster AutoAugment is an offline method, so first search then train CNN models.

### Search

```
python search.py [data.name={cifar10,cifar100,svhn}] [...]
```

This script will save the obtained policy at `policy_weight/DATASET_NAME/EPOCH.pt`.

### Train

```
python train.py path=PATH_TO_POLICY_WEIGHT [data.name={cifar10,cifar100,svhn}] [model.name={wrn28_2,wrn40_2,wrn28_10}]  [...]
```

When `path` is not specified, training is executed without policy, which can be used as a baseline.

## Notice

The codebase here is not exactly the same as the one used in the paper.
For example, this codebase does not include the support for `DistributedDataParallel` and the custom `CutOut` kernel.
But, we believe this reimplementation is much simpler, more stable, and easier to extend for your research projects.

## Citation

```bibtex
@inproceesings{hataya2020a,
    title={{Faster AutoAugment: Learning Augmentation Strategies using Backpropagation}},
    author={Ryuichiro Hataya and Jan Zdenek and Kazuki Yoshizoe and Hideki Nakayama},
    year={2020},
    booktitle={ECCV}
}
```
