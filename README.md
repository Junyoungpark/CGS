# Convergent Graph Solvers

The official code repository of Convergent Graph Solvers (CGS) [[paper](https://arxiv.org/pdf/2106.01680.pdf, "CGS Arxiv link")]

### Requirements

- pytorch
- dgl 0.6.1
- openpnm 2.6.1
- hydra - config managemet

### Quick start

1 .To train `CGS` for porous network problems (Section 6.1)

```
python pn_train.py
```

2. To train `CGS` for graph value iteration problems (Section 6.2)

```
python gvi_train.py
```

3. To train `CGS` for graph benchmark problems (Section 6.3)

```
python benchmark_train.py
```

