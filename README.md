# 💳 Credit Scoring Business Understanding

---

## 📜 Basel II Accord: Why Interpretability & Documentation Matter

> **How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?**

The **Basel II Accord** is an international standard that requires financial institutions to maintain enough cash reserves to cover risks incurred by their operations.  
A key part of the accord is the **"Internal Ratings-Based" (IRB) approach**, which allows banks to use their own internal models to calculate credit risk.

### 🧐 Influence on Interpretability

- For regulators to approve a bank's internal model, they must be able to understand how it works.
- A "black box" model (like a complex neural network or a massive gradient boosting tree) might be highly accurate but is unacceptable if the bank cannot explain why it denied a customer credit.
- **Simple models like Logistic Regression are inherently interpretable.** The coefficients directly tell you how much each feature influences the outcome. This makes it easy to explain to regulators and even to the customer.

### 📝 Influence on Documentation

- Basel II mandates **rigorous validation and documentation**.
- Every step of the modeling process—from data selection and cleaning to feature engineering and model choice—must be justified and documented.
- This ensures the model is robust, not based on spurious correlations, and can be audited.
- Our CI/CD pipeline, clear code, and experiment tracking with MLflow are direct answers to this need for a documented, reproducible process.

---

## 🎯 Why Create a Proxy Variable? What Are the Risks?

> **Since we lack a direct "default" label, why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?**

### ❓ Necessity of a Proxy

- A supervised machine learning model requires a target variable (a "label") to learn from.
- In our dataset, we have transaction history, but no column that says "this customer defaulted" or "this customer paid back their loan."
- **Without a target, we cannot train a model to predict it.**
- Therefore, we must create a **proxy variable**—an observable feature that we believe strongly correlates with the true, unobserved behavior (defaulting).
- In this case, we hypothesize that customers who are highly disengaged (low recency, frequency, and monetary value) are more likely to default on a "buy-now-pay-later" loan. This becomes our proxy for "high-risk."

### ⚠️ Potential Business Risks

- **Risk of Inaccuracy (False Positives/Negatives):**  
  Our proxy is an educated guess. A disengaged customer might not be a defaulter; they might just be a careful spender or have temporarily stopped using the service. Conversely, a highly engaged customer could still face financial hardship and default.
- **Denying Credit to Good Customers (False Positives):**  
  If our model incorrectly flags a good, creditworthy customer as "high-risk" based on the proxy, Bati Bank loses potential business and revenue.
- **Granting Credit to Bad Customers (False Negatives):**  
  If our proxy fails to identify a customer who is truly likely to default, the bank will grant them a loan and likely lose money. This is the most direct financial risk.
- **Model Decay:**  
  The relationship between our proxy (disengagement) and actual default behavior might change over time, making our model less accurate.

---

## ⚖️ Trade-offs: Simple vs. Complex Models in Regulated Finance

> **What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, high-performance model (like Gradient Boosting) in a regulated financial context?**

This is a classic trade-off between **performance** and **interpretability**.

<table>
  <thead>
    <tr>
      <th>Feature</th>
      <th>Simple Model<br>(Logistic Regression)</th>
      <th>Complex Model<br>(Gradient Boosting)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>🎯 <b>Performance</b></td>
      <td>Generally lower predictive accuracy. Assumes a linear relationship between features and the outcome.</td>
      <td>Generally higher predictive accuracy. Captures complex, non-linear relationships and interactions between features.</td>
    </tr>
    <tr>
      <td>🔍 <b>Interpretability</b></td>
      <td>High. Easy to explain. Each feature has a coefficient that shows its impact on the odds of default. Crucial for regulators and business stakeholders.</td>
      <td>Low. A "black box." Very difficult to explain why a specific prediction was made. Techniques like SHAP can help, but it's not as direct.</td>
    </tr>
    <tr>
      <td>📑 <b>Regulatory Compliance</b></td>
      <td>Easier. Meets the Basel II requirement for transparent, understandable models.</td>
      <td>Harder. Difficult to get regulatory approval because of its lack of transparency.</td>
    </tr>
    <tr>
      <td>⚙️ <b>Implementation</b></td>
      <td>Simpler and faster to train. Less prone to overfitting.</td>
      <td>More complex to tune (many hyperparameters). Requires more data and computational power. Can easily overfit if not tuned carefully.</td>
    </tr>
    <tr>
      <td>💼 <b>Business Decision</b></td>
      <td>Choose this model when explainability and regulatory approval are paramount, even at the cost of some accuracy. Provides a solid, defensible baseline.</td>
      <td>Choose this model when maximizing predictive power is the absolute top priority and you have a strategy to manage the "black box" problem (e.g., using it as a challenger model or for non-regulatory purposes).</td>
    </tr>
  </tbody>
</table>

---
