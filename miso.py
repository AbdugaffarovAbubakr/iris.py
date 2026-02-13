from sklearn.datasets import load_iris   # 3 ta klass (setosa, versicolor, virginica)
from sklearn.tree import DecisionTreeClassifier, plot_tree

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Datasetni yuklash
iris = load_iris()

X_train = iris.data
y_train = iris.target

# DataFrame yaratish
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Target ustunini qo‘shish
df['target'] = iris.target

# Birinchi 5 ta qatorni ko‘rish
print(df.head())

# Model yaratish va o‘qitish
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Daraxtni chizish (vizualizatsiya)
plt.figure(figsize=(12, 8))
plot_tree(
    model,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True
)
plt.show()