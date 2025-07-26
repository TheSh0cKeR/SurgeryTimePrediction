# Surgery Duration Predictor

A PyTorch-based deep learning model for predicting the total duration of ENT surgeries at Hillel Yaffe Medical Center. The project uses a deep neural network trained on structured clinical data and optimized using Optuna for hyperparameter tuning. A 15% hold-out test set and stratified cross-validation ensure robust evaluation.

---

## 🏁 Project Objectives

- Improve forecast accuracy for surgery durations using machine learning
- Enable more efficient operating room scheduling and resource usage
- Prove the feasibility of AI-assisted decision making in hospital operations

---

## 🧠 Model Summary

- **Architecture**: Fully connected deep neural network
- **Categorical Encoding**: Learnable embeddings
- **Layers**: 3 hidden layers (256 → 128 → 64) with BatchNorm, ReLU, and Dropout
- **Optimizer**: AdamW with CosineAnnealingLR scheduler
- **Loss Function**: MSELoss
- **Tuning**: Optuna (25 trials, 5-fold stratified CV)
- **Evaluation**: MAE, RMSE, R²

---

## 📁 Dataset Overview

- **Source**: ENT surgery records from Hillel Yaffe Medical Center
- **Shape**: ~4,000 rows × ~80 columns
- **Target**: `Total Surgery Time` in minutes
- **Features**:
  - Patient demographics: age, gender, weight, BMI, smoking status
  - Clinical attributes: allergies, blood test results, comorbidities
  - Surgery details: procedure codes, anesthesia type, surgery type

---

## ⚙️ Preprocessing Pipeline

- Drop near-identifier columns (`>90%` unique values)
- Generate count features for multi-valued fields (e.g., drug names)
- Parse and one-hot encode top 40 procedure codes
- Separate and transform:
  - **Categoricals** → Embedding via factorized mapping
  - **Numerics** → Scaled using `StandardScaler`, low-variance numerics removed
- Missing values handled:
  - Categorical: filled with "UNK"
  - Numeric: filled with median

---

## 🧪 Evaluation Strategy

- Stratified 85/15 train-test split using quartiles of surgery time
- 5-fold CV with early stopping (patience = 15)
- Final evaluation on 15% held-out test set

### 📊 Final Results (Test Set)

| Metric      | Value     |
|-------------|-----------|
| MAE         | ~9.70 min |
| RMSE        | ~15.24    |
| R² Score    | ~0.928    |

---

## 🗃️ Directory Structure

```
surgery-time-predictor/
├── data/                      # Input CSV dataset
│   └── Surgery Dataset.csv
├── outputs/                   # Model checkpoints and metadata
│   ├── best_surgery_model_v4.pt
│   └── best_model_info_v4.txt
├── src/
│   └── Model.py               # Main training pipeline
├── requirements.txt
└── README.md
```

---

## ▶️ Run the Model

1. **Install Python packages**:

```bash
pip install -r requirements.txt
```

2. **Place the dataset** in the correct path:

```bash
data/SurgeryDataset.csv
```

3. **Run the training script**:

```bash
python src/Model.py
```

Model artifacts (trained model + metadata) will be saved to the `outputs/` folder.

---

## 📚 References

- Riahi, V., et al. (2023). *Improving Preoperative Prediction of Surgery Duration*. BMC Health Services Research.
- Caserta, M., & García Romero, A. (2024). *A Novel Approach to Forecast Surgery Durations*. Springer.

---

