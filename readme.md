# Blockchain Fraud Detection using Graph Neural Networks

This project implements Graph Convolutional Network (GCN) models to detect fraudulent Bitcoin transactions using the Elliptic dataset. The dataset maps Bitcoin transactions (nodes) to their corresponding classes (licit or illicit), and edges represent Bitcoin flows between transactions.

## Dataset

The Elliptic dataset consists of:
- **nodes.csv**: Bitcoin transactions with features and labels (licit or illicit)
- **edges.csv**: Transaction flows connecting the nodes

Place these files in the `data/raw/` directory.

## Project Structure

```
blockchain-fraud-detection/
├── data/
│   ├── raw/                     # Original dataset files
│   └── processed/               # Processed datasets and features
├── notebooks/                   # Jupyter notebooks
│   ├── 1_data_exploration.ipynb
│   ├── 2_feature_engineering.ipynb
│   ├── 3_model_development.ipynb
│   ├── 4_model_evaluation.ipynb
│   └── 5_fraud_case_study.ipynb
├── src/                         # Source code modules
│   ├── data_preparation.py
│   ├── feature_extraction.py
│   ├── gnn_model.py
│   ├── train.py
│   ├── evaluate.py
│   └── visualization.py
├── models/                      # Saved models
└── reports/                     # Analysis reports and visuals
    └── figures/
```

## Getting Started

1. Install requirements:
   ```
   pip install -r requirements.txt
   ```

2. Place the Elliptic dataset files (`nodes.csv` and `edges.csv`) in the `data/raw/` directory.

3. Run the notebooks in sequence:
   - Start with data exploration
   - Feature engineering
   - Model development 
   - Evaluation
   - Case study analysis

## Key Features

- **Graph-based Fraud Detection**: Utilizing graph structure to identify fraudulent transactions
- **Advanced Feature Engineering**: Combining transaction attributes with graph features
- **GCN Model Architecture**: Multi-layer GCN with residual connections and batch normalization
- **Multiple Data Splitting Strategies**: Time-based, random, and community-based approaches
- **Comprehensive Evaluation**: Using classification metrics, ROC curves, and visualization tools

## Results

The GCN model achieves strong performance in detecting fraudulent transactions with key metrics:
- High precision and recall for fraud detection
- Effective utilization of graph structure
- Interpretable patterns in fraud detection

## License

This project is licensed under the MIT License - see the LICENSE file for details.