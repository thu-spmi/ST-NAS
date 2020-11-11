# ST-NAS on ASR

## Get started

Clone the source code from GitHub:

```
$ git clone https://github.com/thu-spmi/ST-NAS
```

### Requirements and Install

- `Python>3`
- `PyTorch>=1.0.0`. Recommend to use `PyTorch>1.1.0` due to a non-critical bug of `PyTorch<=1.1.0`, which won't affect the perfermance of ST-NAS, but would consume more GPU memory in evaluation.

This work is based on the end-to-end  ASR platform [CAT](https://github.com/thu-spmi/CAT).

1. To start with, please firstly install CAT.
2. For the WSJ task, afer installing CAT,  create a new `egs/wsj_NAS`:

```
$ cp -r CAT/egs/wsj CAT/egs/wsj_NAS
```

3. Copy the patch files:

```
$ cp ST-NAS/ASR/egs/wsj_NAS/* CAT/egs/wsj_NAS/
```

4. Link the scripts of ST-NAS:

```
$ ln -s <absolute path to ST-NAS>/ASR/scripts CAT/egs/wsj_NAS/scripts
```

5. There are still some required packages to manually install, most of which could be easily installed with `pip` or `conda`.

## Prepare Data

```
$ cd CAT/egs/wsj_NAS
$ ./preprocess.sh
```

Note that `./preprocess.sh` is mostly borrowed from `CAT/egs/wsj/run.sh`, but the subsampling configuration is removed.

## Search and Validate

ST-NAS is divided into three stages: warmup, architecture search and retrain.

```
$ cd CAT/egs/wsj_NAS
$ ./nastrain.sh
```

After searching (including all three stages) finishing, run

```
$ cd CAT/egs/wsj_NAS
$ ./validate.sh
```

To visualize the searched architecture

```
$ cd CAT/egs/wsj_NAS
$ python3 scripts/ArchSearch.py --visualize --nasConfig="./config.json" --resume="models/as/model.bestforretrain.pt"
```

`graphviz` package is required. Follow the instruction [here](https://pypi.org/project/graphviz/) to install it.
