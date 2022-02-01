# Convergent Graph Solvers

The official code repository of Convergent Graph Solvers (CGS)

## Paper
For more details, please see our paper [Convergent Graph Solvers](https://openreview.net/forum?id=ItkxLQU01lD) which has been accepted at ICLR 2022. 
If this code is useful for your work, please consider to cite our paper:
```
@inproceedings{
    park2022convergent,
    title={Convergent Graph Solvers},
    author={Junyoung Park and Jinhyun Choo and Jinkyoo Park},
    booktitle={International Conference on Learning Representations},
    year={2022},
    url={https://openreview.net/forum?id=ItkxLQU01lD}
}
```


## Requirements

- pytorch
- dgl 0.7.0
- openpnm 2.8.2
- hydra 1.1

## Quick start

1 .To train `CGS` for porous network problems (Section 6.1)

```console
python pn_train.py
```

2. To train `CGS` for graph value iteration problems (Section 6.2)

```console
python gvi_train.py
```

3. To train `CGS` for graph benchmark problems (Section 6.3)

```console
python benchmark_train.py
```



