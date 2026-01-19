# Water Quality Prediction Pipeline (AKH Dataset)

This project provides a professional-grade machine learning framework to classify water quality based on physicochemical parameters. It compares three distinct architectures: **Random Forest**, **XGBoost**, and **Long Short-Term Memory (LSTM)**.

---

## 1. Scientific Context & Data Transformation

### The Water Quality Index (WQI) Equation
The ground truth labels in this project are derived from the **Weighted Arithmetic Water Quality Index** method. The scientific formula used to generate the initial scores is:

$$WQI = \frac{\sum Q_i W_i}{\sum W_i}$$

Where:
* **$Q_i$ (Quality Rating):** Calculated for each parameter as:
  $$Q_i = 100 \times \left( \frac{V_i - V_{ideal}}{S_i - V_{ideal}} \right)$$
* **$W_i$ (Unit Weight):** Inversely proportional to the standard permissible value:
  $$W_i = \frac{K}{S_i} \text{ where } K = \frac{1}{\sum (1/S_i)}$$
* **$S_i$:** Standard permissible value for the $i^{th}$ parameter.
* **$V_i$:** Observed value of the $i^{th}$ parameter.

### Label Classification
The continuous WQI score is discretized into the following target classes ($y$):
* **Class 0 (Excellent):** $WQI \leq 25$
* **Class 1 (Good):** $25 < WQI \leq 50$
* **Class 2 (Bad):** $WQI > 50$

### Pre-processing: Standardization
To ensure features with large ranges (like Coliforms) do not mathematically dominate features with small ranges (like PH), we apply **Z-score Normalization**:
$$z = \frac{x - \mu}{\sigma}$$

---

## 2. How the Machine Learning Models Work

### A. Random Forest (RF): The Ensemble of Experts
Random Forest is a "bagging" algorithm that builds a committee of 100+ independent decision trees.
* **The Learning Logic:** It uses **Recursive Partitioning**. At each node of a tree, it calculates the **Gini Impurity ($G$)** to determine the most effective chemical threshold to split the data:
  $$G = 1 - \sum_{i=1}^{C} (p_i)^2$$
  *(Where $p_i$ is the probability of a sample belonging to class $i$)*.
* **Mechanism:** Every tree casts a vote for a class. The final prediction is the class with the most votes. This reduces **variance** and prevents the model from being distracted by outliers.



### B. XGBoost: The Sequential Perfectionist
XGBoost (Extreme Gradient Boosting) builds trees sequentially, where each new tree is designed to fix the mistakes of the previous ones.
* **The Learning Logic:** It minimizes a regularized **Objective Function** that balances predictive power with model simplicity:
  $$\text{Obj}(\theta) = \sum_{i=1}^{n} L(y_i, \hat{y}_i) + \sum_{k=1}^{K} \Omega(f_k)$$
  * $L$ is the loss function (the error).
  * $\Omega$ is the penalty for model complexity (Regularization).
* **Mechanism:** It calculates the "Gradient" (direction of error) and moves the model's weights in the opposite direction to find the global minimum error.



### C. LSTM: The Historical Memory
LSTM (Long Short-Term Memory) is a Deep Learning architecture (RNN) designed to recognize patterns in sequences (time-series).
* **The Learning Logic:** It maintains a **Cell State ($C_t$)** that acts as a memory buffer. It manages this memory via three mathematical gates using the Sigmoid ($\sigma$) and Tanh functions:
  1. **Forget Gate ($f_t$):** $\sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$ — Deletes irrelevant past data.
  2. **Input Gate ($i_t$):** $\sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$ — Saves important current data.
  3. **Output Gate ($o_t$):** $\sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$ — Filters the memory to make a prediction.
* **Mechanism:** This is ideal for water data collected over time, as it understands that a pollution spike 2 days ago influences the quality today.



---

## 3. Interpreting Feature Importance
The pipeline generates a **Feature Importance Plot**. This provides the scientific "Why" behind the model's decisions:
* **High Importance (e.g., DO, BOD5):** Indicates these are the primary drivers of the water quality index in your dataset.
* **Low Importance (e.g., Temp):** Suggests these variables are less predictive for the final classification, even if they are physically present.

---

## 4. Overall Model Comparison

| Feature | Random Forest | XGBoost | LSTM |
| :--- | :--- | :--- | :--- |
| **Philosophy** | Parallel/Voting | Sequential/Correction | Memory/Sequences |
| **Mathematics** | Gini Impurity | Gradient Descent | Gated Tensors |
| **Interpretability** | Very High | High | Low (Black Box) |
| **Training Speed** | Fast | Very Fast | Slow (Deep Learning) |
| **Data Context** | Each row is independent | Each row is independent | Rows are linked in time |

---

## 5. Setup & Installation

1. **Prepare Data:** Ensure `AKH_WQI.csv` is located in the `data/` directory.
2. **Install Requirements:**
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn xgboost torch