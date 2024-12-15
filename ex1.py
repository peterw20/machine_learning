from sklearn.datasets import load_breast_cancer
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = load_breast_cancer()
X = data.data
y = data.target
print(f"Dataset Loaded: {X.shape[0]} samples, {X.shape[1]} features, {len(set(y))} target classes")

tsne = TSNE(n_components=2, random_state=42, perplexity=30) 
X_subset = X[:200] 
y_subset = y[:200]
X_tsne = tsne.fit_transform(X_subset)

tsne_df = pd.DataFrame(X_tsne, columns=['TSNE Component 1', 'TSNE Component 2'])
tsne_df['Target'] = y_subset

plt.figure(figsize=(8, 6))
sns.scatterplot(data=tsne_df, x='TSNE Component 1', y='TSNE Component 2', hue='Target', palette='viridis')
plt.title('TSNE Visualization ')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train/Test Split: {X_train.shape[0]} train samples, {X_test.shape[0]} test samples")

model_decision_tree = DecisionTreeClassifier(random_state=40, max_depth=4, min_samples_split=9)
model_decision_tree.fit(X_train, y_train)

model_random_forest = RandomForestClassifier(random_state=35, n_estimators=120, max_depth=9, min_samples_split=4)
model_random_forest.fit(X_train, y_train)

model_adaboost = AdaBoostClassifier(random_state=35, n_estimators=95, learning_rate=0.8)
model_adaboost.fit(X_train, y_train)


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f"\n{name} Results:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    
    return accuracy, precision, recall, f1

results = {}
results['Decision Tree'] = evaluate_model("Decision Tree", model_decision_tree, X_test, y_test)
results['Random Forest'] = evaluate_model("Random Forest", model_random_forest, X_test, y_test)
results['AdaBoost'] = evaluate_model("AdaBoost", model_adaboost, X_test, y_test)

best_model = max(results.items(), key=lambda x: x[1][3]) 
print(f"\nThe best model is {best_model[0]} with an F1 score of {best_model[1][3]:.2f}")
