import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from matplotlib.colors import ListedColormap


df = pd.read_csv("Coherence_bert_cls_embeddings.csv")

df['embedding'] = df['embedding'].apply(ast.literal_eval)

df['f1'] = df['embedding'].apply(lambda x: x[0])
df['f2'] = df['embedding'].apply(lambda x: x[1])

target = "label"

def calculate_entropy(data):
    values, counts = np.unique(data, return_counts=True)
    probabilities = counts / counts.sum()
    
    entropy = 0
    for p in probabilities:
        entropy -= p * np.log2(p)
    
    return entropy


def equal_width_binning(data, bins=4):
    
    min_val = np.min(data)
    max_val = np.max(data)
    
    width = (max_val - min_val) / bins
    
    binned = np.floor((data - min_val) / width)
    binned[binned == bins] = bins - 1
    
    return binned.astype(int)


entropy_value = calculate_entropy(df[target])

print("A1 RESULT")
print("Entropy of Dataset:", entropy_value)


def calculate_gini(data):
    
    values, counts = np.unique(data, return_counts=True)
    
    probabilities = counts / counts.sum()
    
    gini = 1 - np.sum(probabilities ** 2)
    
    return gini


gini_value = calculate_gini(df[target])

print("\nA2 RESULT")
print("Gini Index:", gini_value)


def information_gain(data, feature, target):
    
    total_entropy = calculate_entropy(data[target])
    
    values, counts = np.unique(data[feature], return_counts=True)
    
    weighted_entropy = 0
    
    for v, c in zip(values, counts):
        
        subset = data[data[feature] == v]
        
        entropy_subset = calculate_entropy(subset[target])
        
        weighted_entropy += (c / np.sum(counts)) * entropy_subset
        
    ig = total_entropy - weighted_entropy
    
    return ig


def find_root_node(data, features, target):
    
    ig_values = {}
    
    for feature in features:
        ig_values[feature] = information_gain(data, feature, target)
    
    root = max(ig_values, key=ig_values.get)
    
    return root, ig_values

df["f1_bin"] = equal_width_binning(df["f1"], bins=4)
df["f2_bin"] = equal_width_binning(df["f2"], bins=4)

features = ["f1_bin", "f2_bin"]

root, ig_values = find_root_node(df, features, target)

print("\nA3 RESULT")
print("Information Gain Values:", ig_values)
print("Root Node Feature:", root)

def binning(data, bins=4, method="width"):
    
    if method == "width":
        return equal_width_binning(data, bins)
    
    elif method == "frequency":
        return pd.qcut(data, bins, labels=False)
    
    else:
        raise ValueError("Invalid binning method")


df["f1_bin"] = binning(df["f1"], bins=4, method="width")

print("\nA4 RESULT")
print(df[["f1", "f1_bin"]].head())


class MyDecisionTree:
    
    def __init__(self):
        self.tree = None
        
    def build_tree(self, data, features, target):
        
        if len(np.unique(data[target])) == 1:
            return data[target].iloc[0]
        
        if len(features) == 0:
            return data[target].mode()[0]
        
        root, _ = find_root_node(data, features, target)
        
        tree = {root: {}}
        
        for value in np.unique(data[root]):
            
            subset = data[data[root] == value]
            
            remaining_features = [f for f in features if f != root]
            
            subtree = self.build_tree(subset, remaining_features, target)
            
            tree[root][value] = subtree
        
        return tree
    
    def fit(self, data, features, target):
        self.tree = self.build_tree(data, features, target)


my_tree = MyDecisionTree()

my_tree.fit(df, features, target)

print("\nA5 RESULT")
print("Custom Decision Tree:", my_tree.tree)



X = df[["f1", "f2"]]
y = df[target]

model = DecisionTreeClassifier()

model.fit(X, y)

plt.figure(figsize=(10,6))

plot_tree(
    model,
    feature_names=["f1","f2"],
    class_names=True,
    filled=True
)

plt.title("Decision Tree Visualization")

plt.show()


x_min, x_max = X["f1"].min() - 1, X["f1"].max() + 1
y_min, y_max = X["f2"].min() - 1, X["f2"].max() + 1

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.01),
    np.arange(y_min, y_max, 0.01)
)

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8,6))

plt.contourf(xx, yy, Z, alpha=0.4)

plt.scatter(X["f1"], X["f2"], c=y)

plt.xlabel("Feature 1 (Embedding)")
plt.ylabel("Feature 2 (Embedding)")

plt.title("Decision Boundary using Decision Tree")

plt.show()
