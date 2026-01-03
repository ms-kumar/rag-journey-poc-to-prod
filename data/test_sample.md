# Machine Learning Fundamentals

Machine learning is a branch of artificial intelligence focused on building systems that learn from data.

## Core Concepts

### Supervised Learning
Supervised learning uses **labeled data** to train models. Common tasks include:
- Classification
- Regression
- Time series prediction

### Unsupervised Learning
Works with unlabeled data to find patterns:
- Clustering (K-means, DBSCAN)
- Dimensionality reduction (PCA, t-SNE)
- Anomaly detection

### Deep Learning
Neural networks with multiple layers that can learn hierarchical representations.

## Applications
1. Natural Language Processing
2. Computer Vision
3. Recommendation Systems
4. Autonomous Vehicles

## Code Example

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

## Key Takeaways
- ML enables computers to learn without explicit programming
- Different algorithms suit different problem types
- Data quality is crucial for model performance
