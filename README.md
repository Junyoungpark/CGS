# Convergent Graph Solvers

The official code repository of Convergent Graph Solvers (CGS) [[paper](https://arxiv.org/abs/2106.01680)]

### Requirements

- pytorch
- dgl 0.7.0
- openpnm 2.8.2
- hydra 1.1

### Quick start

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

