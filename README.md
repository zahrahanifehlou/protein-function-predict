# Drug–Target Binding Affinity Prediction

This project explores multiple machine learning approaches to predict the 
binding affinity between drug molecules and protein targets.

The goal is to build, compare, and analyze classical ML methods, deep learning models,
and modern representation learning approaches used in drug discovery.

---

## 🔬 Problem Definition

Given:
- Drug represented as a SMILES string
- Target represented as an amino acid sequence

Predict:
- Binding affinity (e.g., Kd, Ki, IC50, or pKd)

---

## 📊 Datasets
- BindingDB (subset)
    - Dataset shape: (1645667, 6)
    - unique targets: 8203
    - unique SMILES: 1037995
    - Binding affinity range (pKi): -5.456427145044637 to 20.0
    - train samples size: 1316535

Each dataset includes:
- SMILES
- Protein sequence
- Binding affinity value

---

## 🧠 Methods Implemented

### 1️⃣ Classical Machine Learning
- Random Forest
- XGBoost
- Support Vector Regression
- Feature-based representations:
  - Morgan fingerprints
  - AAC / k-mer protein features

### 2️⃣ Deep Learning
- CNN on SMILES + protein sequences (DeepDTA-style)
- LSTM / BiLSTM models
- Graph Neural Networks (GraphDTA)

### 3️⃣ Recent & Modern Approaches
- Transformer-based protein encoders (ESM / ProtBERT)
- SMILES Transformers
- Multimodal fusion networks
- Pretrained embeddings + regression head

---

## 📈 Evaluation Metrics

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Concordance Index (CI)
- Pearson Correlation

---

## 🧪 Key Experiments

| Model | Representation | RMSE | CI |
|-----|---------------|------|----|
| Random Forest | Morgan + AAC | 0.92 | 0.74 |
| CNN-DTA | Sequence-based | 0.78 | 0.82 |
| GNN-DTA | Molecular graph | 0.73 | 0.85 |
| Transformer-DTA | Pretrained embeddings | **0.68** | **0.88** |

---

## 🧠 What I Learned

- Trade-offs between interpretability and performance
- Importance of representation learning in bioinformatics
- How pretrained biological models improve downstream tasks
- Reproducible ML experimentation

---

## 🚀 Future Work

- Multi-task learning
- Knowledge Graph integration
- Uncertainty estimation
- Zero-shot target prediction

---

## 👩‍💻 Author

**Zahra**  
Aspiring AI / Data Scientist  
Focus: AI for Drug Discovery & Bioinformatics
