import pandas as pd
from sklearn.datasets import load_iris

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

