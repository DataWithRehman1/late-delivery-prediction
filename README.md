# Late Delivery Prediction — E-Commerce

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)

> **Predict whether an e-commerce order will arrive late *before shipment*, so that high-risk orders can be prioritised operationally.**

---

## Table of Contents
- [Business Problem](#business-problem)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Approach](#approach)
- [Model Performance](#model-performance)
- [Quick Start](#quick-start)
- [Batch Prediction Pipeline](#batch-prediction-pipeline)

---

## Business Problem

Late deliveries hurt customer satisfaction scores and increase customer-care call volume. By scoring orders *before* they ship, the logistics team can:

- Flag high-risk orders for **Priority Shipment**
- Pre-emptively contact customers
- Reduce operational fire-fighting

---

## Dataset

| Property | Value |
|---|---|
| Source | E-Commerce Shipping Dataset |
| Rows | 10,999 |
| Features | 11 (after dropping ID) |
| Target | `Reached.on.Time_Y.N` — `1` = Late, `0` = On-Time |
| Class balance | ~60% Late / ~40% On-Time |

**Features:**

| Feature | Type | Description |
|---|---|---|
| `Warehouse_block` | Categorical | Warehouse section (A–F) |
| `Mode_of_Shipment` | Categorical | Flight / Ship / Road |
| `Customer_care_calls` | Numeric | Number of calls before dispatch |
| `Customer_rating` | Numeric | 1–5 rating |
| `Cost_of_the_Product` | Numeric | USD |
| `Prior_purchases` | Numeric | Historical order count |
| `Product_importance` | Categorical | Low / Medium / High |
| `Gender` | Categorical | M / F |
| `Discount_offered` | Numeric | Discount percentage |
| `Weight_in_gms` | Numeric | Product weight |

---

## Project Structure

```
late-delivery-prediction/
├── e_commerce.ipynb          # Full analysis: EDA → Feature Engineering → Modelling
├── predict.py                # Batch prediction script
├── Train.csv                 # Training dataset
├── new_orders.csv            # Example input for batch predictions
├── prediction_results.csv    # Example output from predict.py
├── late_delivery_model.pkl   # Serialised trained model
├── requirements.txt          # Python dependencies
└── README.md
```

---

## Approach

1. **EDA** — Explored target distribution, shipment modes, and confirmed zero missing values.
2. **Feature Engineering** — Dropped `ID`, applied one-hot encoding to categorical columns.
3. **Baseline Model** — Logistic Regression with default settings (accuracy ≈ 65%).
4. **Business-Tuned Model** — Increased the penalty for misclassifying Late orders via `class_weight={0: 1, 1: 2}` to maximise Late-delivery recall.

---

## Model Performance

### Baseline Logistic Regression

| Class | Precision | Recall | F1 |
|---|---|---|---|
| On-Time (0) | 0.57 | 0.57 | 0.57 |
| **Late (1)** | 0.70 | **0.70** | 0.70 |
| Accuracy | | **64.7%** | |

### Tuned Model (`class_weight={0: 1, 1: 2}`)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| On-Time (0) | 0.64 | 0.08 | 0.15 |
| **Late (1)** | 0.61 | **0.97** | 0.75 |
| Accuracy | | **61.0%** | |

> **Decision:** We intentionally sacrifice On-Time recall to achieve **97% Late-delivery recall**. Missing a late shipment costs more than a false alarm — the operations team can verify flagged orders quickly, but a missed late delivery is unrecoverable.

---

## Quick Start

### 1. Clone & install dependencies

```bash
git clone https://github.com/DataWithRehman1/late-delivery-prediction.git
cd late-delivery-prediction
pip install -r requirements.txt
```

### 2. Train the model (optional — pre-trained model included)

Open and run `e_commerce.ipynb` end-to-end. The notebook saves `late_delivery_model.pkl`.

### 3. Run batch predictions

```bash
python predict.py
```

Reads `new_orders.csv`, scores every order, and writes results to `prediction_results.csv`.

---

## Batch Prediction Pipeline

**Input** (`new_orders.csv`) — same columns as training data, without the target column:

```
ID,Warehouse_block,Mode_of_Shipment,...
1001,C,Ship,...
```

**Output** (`prediction_results.csv`):

```
order_id,prediction,prediction_label,action
1001,1,Late,Priority Shipment
1002,0,On-Time,Normal Shipment
```

---

## License

[MIT](LICENSE)
