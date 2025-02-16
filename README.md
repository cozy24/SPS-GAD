# SPS-GAD

## Overview

SPS-GAD is a novel framework for graph anomaly detection that addresses the challenges of heterophily in graph-structured data. The method incorporates a subgraph sparsification module, a hybrid spectral filter, and an attention mechanism to enhance anomaly detection performance. SPS-GAD handles heterophilic graphs by classifying subgraphs into homophilic, heterophilic, and ambiguous categories, applying tailored spectral filters for each subgraph type, and learning both spatial and spectral node representations.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/SPS-GAD.git
   cd SPS-GAD
   ```

2. **Install dependencies:**

   You can install the required Python packages using `pip`:

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your environment:**

   Ensure you have the necessary dependencies for dataset loading and processing (e.g., DGL, PyTorch). Follow the DGL and PyTorch installation instructions based on your system.

## Usage

### Command-line arguments

You can run the model by executing the following command:

```bash
python main.py --dataset yelp
```

### Arguments

- `--dataset` (str): Specify the dataset to use. Available options include `yelp`, `weibo`, `amazon`, `elliptic`, and `tolokers`.
- `--run` (int): The number of runs for the experiment. Default is 10.
- `--epoch` (int): Number of epochs to train the model. Default is 1000.
- `--patience` (int): Patience for early stopping. The model will stop training if there is no improvement for this many epochs. Default is 100.
- `--order` (int): The order of polynomial convolution. . Default is 2.
- `--homo` (int): Set this to 1 for a homophilic graph, or 0 for a heterophilic graph. This influences the subgraph partitioning.
- `--pos_quantile` (float): The quantile for positive nodes in the dataset. Default is 0.3.
- `--neg_quantile` (float): The quantile for negative nodes in the dataset. Default is 0.3.
- `--num_heads` (int): The number of attention heads to use in the GNN model. Default is 1.

### Example

To run the model with the `yelp` dataset, 10 runs, 1000 epochs, patience of 100, with a homophilic graph and specific quantile settings, execute:

```bash
python main.py --dataset yelp --run 10 --epoch 1000 --patience 100 --homo 1 --pos_quantile 0.3 --neg_quantile 0.3 --num_heads 1
```

This will train the SPS-GAD model using the specified hyperparameters and dataset.

### Output

The results of the experiment will be stored in the `output` directory by default. This will include:

- Model checkpoints
- Training logs
- Performance metrics (e.g., F1-Macro, AUC, G-Mean)

## Model Description

SPS-GAD works by first reconstructing node features and then applying a hybrid spectral filter to capture essential information about the underlying graph structure. The model introduces an edge partitioner to classify subgraphs into homophilic, heterophilic, and ambiguous categories, applying different spectral filters based on the subgraph type. Additionally, it incorporates a weighted loss function to address label imbalance during training.

For further details on the model architecture and experiments, please refer to the paper.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
