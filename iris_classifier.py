import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

iris = load_iris()
print(iris.keys())
print("feature names", iris.feature_names)
print("target names", iris.target_names)

# Create a DataFrame from the iris dataset
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
# Add the target column to the DataFrame
df['target'] = iris.target

print(df.head())
print(df.info())
print(df.describe())

sns.pairplot(df, hue='target', palette='husl')
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

x = df.drop('target', axis=1)
y = df['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print("Training set size:", x_train.shape)
print("Test set size:", x_test.shape)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)
print("Model trained successfully.")

y_pred = knn.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()