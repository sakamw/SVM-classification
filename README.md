# Heart Disease Prediction — SVM Classifier

A machine learning solution to predict the presence of heart disease using a **Support Vector Machine (SVM)** on the UCI Cleveland Heart Disease dataset.

---

## Dataset

| Source | Link                                                                                                                           |
| ------ | ------------------------------------------------------------------------------------------------------------------------------ |
| Kaggle | [kaggle.com/datasets/cherngs/heart-disease-cleveland-uci](https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci) |

| Detail       | Value                                   |
| ------------ | --------------------------------------- |
| Patients     | 303                                     |
| Features     | 13 clinical features                    |
| Target       | `1` = Disease Present, `0` = No Disease |
| Disease rate | ~54%                                    |

---

## Algorithm: Support Vector Machine (SVM)

### Why SVM?

| Reason                  | Explanation                                                                     |
| ----------------------- | ------------------------------------------------------------------------------- |
| **Small dataset**       | 303 rows — SVM generalises well without needing thousands of samples            |
| **RBF Kernel**          | Maps features into higher-dimensional space for non-linear decision boundaries  |
| **Margin maximisation** | Robust to noisy/outlier medical data — only support vectors define the boundary |
| **Balanced classes**    | ~54% vs 46% split — SVM handles this naturally without oversampling             |

### How It Works (Intuition)

```
      ●  ●  ●                  ← Healthy patients
        ●  ●
  ─ ─ ─ ─ ─ ─ ─ ─ ─          ← Decision hyperplane
        ●  ●
      ●  ●  ●                  ← Diseased patients

      ↕ Margin is maximised ↕
```

SVM finds the _widest possible gap_ between the two classes. Only the boundary points (**support vectors**) determine the hyperplane — making it robust to noise.

### Hyperparameters (Tuned via GridSearchCV)

| Parameter | What It Controls                                                              |
| --------- | ----------------------------------------------------------------------------- |
| `C`       | Regularisation — small C = wider margin, large C = fits training data tightly |
| `gamma`   | RBF kernel bandwidth — how far each training point's influence reaches        |
| `kernel`  | `rbf` (non-linear) vs `linear` — best selected automatically                  |

---

## Results

| Metric        | Score        |
| ------------- | ------------ |
| ROC-AUC       | ~0.92        |
| PR-AUC        | ~0.91        |
| 5-Fold CV AUC | ~0.90 ± 0.04 |

---

## Requirements

```
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
```
