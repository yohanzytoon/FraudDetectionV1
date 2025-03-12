# Bitcoin Transaction Fraud Detection using Graph Neural Networks

This repository contains a comprehensive solution for detecting fraudulent transactions in Bitcoin networks using Graph Neural Networks (GNNs). The implementation leverages the graph structure of transaction networks alongside traditional transaction features to improve fraud detection accuracy.

## Project Overview

Cryptocurrency fraud detection presents unique challenges due to the pseudonymous nature of transactions and the complex network structures that can obscure fraudulent activities. This project implements a multi-stage approach to detect fraud in Bitcoin transaction networks by:

1. **Processing raw transaction data** into a suitable format for graph-based analysis
2. **Engineering graph-based features** that capture transactional relationships
3. **Training various GNN architectures** to learn patterns of fraudulent behavior
4. **Evaluating and comparing model performance** with comprehensive metrics
5. **Visualizing transaction patterns** to provide insights into network structures


## Models Implemented

The project implements and compares three state-of-the-art GNN architectures:

1. **Graph Convolutional Network (GCN)** - Learns node representations through neighborhood aggregation
2. **GraphSAGE** - Samples and aggregates features from node neighborhoods for scalable inductive learning
3. **Graph Attention Network (GAT)** - Leverages attention mechanisms to weigh the importance of neighboring nodes

## Performance Results

Based on test set evaluation, the GNN models achieved the following performance metrics:

| Model | Accuracy | F1-Score | ROC AUC | PR AUC |
|-------|----------|----------|---------|--------|
| GCN   | 0.8968   | 0.8540   | 0.6842  | 0.1587 |
| SAGE  | 0.9021   | 0.8559   | 0.8080  | 0.3576 |
| GAT   | 0.8550   | 0.8327   | 0.5129  | 0.0918 |

The GraphSAGE model demonstrated the best overall performance, particularly in ROC AUC and PR AUC metrics, making it well-suited for the imbalanced nature of fraud detection tasks.

## Installation and Setup

1. Clone the repository:
```bash
git clone https://github.com/yohanzytoon/FraudDetectionV1.git
cd FraudDetectionV1
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare the data:
```bash
python prepare_data.py
```

4. Run the data processing pipeline:
```bash
python data_preparation.py
python feature_extraction.py
```

5. Train models:
```bash
python train.py
```

6. Evaluate models:
```bash
python evaluate.py
```

7. Generate visualizations:
```bash
python visualization.py
```

## Data Requirements

The system expects input data in the following format:
- `classes.csv`: Transaction IDs and their labels (legitimate/fraudulent)
- `edgelist.csv`: Source and target transaction IDs representing edges
- `Features.csv`: Transaction features for each transaction ID

## Customization

The implementation is designed to be modular, allowing for easy customization:
- Modify hyperparameters in `train.py` to optimize for specific datasets
- Add new GNN architectures in `gnn_model.py`
- Extend feature engineering in `feature_extraction.py`

## Key Insights

Our analysis revealed several important patterns in fraudulent transactions:
- Fraudulent transactions tend to exhibit distinctive connectivity patterns
- Graph-based features such as centrality measures provide strong signals for fraud detection
- The temporal distribution of transactions offers additional discriminative power for identifying suspicious activities

## Future Work

- Integration of more advanced graph sampling techniques for larger networks
- Exploration of heterogeneous graph models to incorporate transaction types
- Development of real-time fraud detection capabilities
- Investigation of explainable AI methods for fraud detection

