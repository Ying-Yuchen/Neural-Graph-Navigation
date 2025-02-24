# Neural-Graph-Navigation
the code of the paper : Neural Graph Navigation for Efficient Subgraph Matching


# **Overview**

```
method/
├── README.md
├── NeuGN/
│   ├── encoders/
│   │   ├── mpnn_encoder.py (GCN, GIN)
│   │   ├── nag_encoder.py (NAGphormer)
│   ├── graph_tokenizer.py (as the name suggests)
│   ├── datasets.py (process for datasets)
│   ├── nx_utils.py (contains a series of NetworkX utilities for obtaining Eulerian paths)
│   ├── utils.py (contains some data processing functions)
│   ├── model.py (NeuGN related)
├── main.py
```

Subgraph Matching Dataset: See ./datasets/subgraphmatching

## **Training Setup**

- Create the necessary folders and set the parameters (see ./data_params/wikics/model_args.yaml for details).
- Just run the code.

```bash
Example:
python -m torch.distributed.launch --nproc_per_node 4 main.py --config ./model_params/wikics--load_params 0
```