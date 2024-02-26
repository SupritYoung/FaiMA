This is a Pytorch implementation and released dataset of "FaiMA: Feature-aware In-context Learning for Multi-Domain Aspect-based Sentiment Analysis" accepted by LREC-COLING 2024.

# Feature-aware In-context Learning for Multi-Domain Aspect-based Sentiment Analysis (FaiMA)

More details of the paper and dataset will be released after it is published.

# The Code

## Requirements

Following is the suggested way to install the dependencies:

    pip install -r requirements.txt


## Folder Structure

```tex
└── SA-LLM
    ├── data                    # Contains the datasets
    │   ├── inst/ASPE           # Our MD-ASPE instruction data
    │   ├── raw/ASPE            # MD-ASPE raw data
    ├── checkpoints             # Contains the trained checkpoint for model weights
    ├── src
    │   ├── gnnencoder          # The code related to MGATE
    │   ├── Icl                 # The code related to Feature-aware In-context Learning
    │   ├── llmtuner            # The code related to LLM train, predict etc.
    ├── run_gnn.py              # The code for training MGATE
    ├── run_aspe.py             # The code for training FaiMA and baselines
    └── README.md               # This document
```

##  Training and Evaluation

[//]: # (首先运行 `run_gnn.py` 训练 MGATE 模型，然后运行 `run_aspe.py` 训练 FaiMA 模型。)

1. Run `run_gnn.py` to train MGATE model.
2. Run `run_aspe.py` to train FaiMA and baselines, replece `model_name_or_path` with your llama model weight path.
