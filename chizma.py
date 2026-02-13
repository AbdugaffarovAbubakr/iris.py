import numpy as np
from sklearn.linear_model import LinearRegression

# Ma’lumotlar
x = np.array([2, 4, 6, 8]).reshape(-1, 1)
y = np.array([55, 65, 75, 85])

# Model qurish
model = LinearRegression()
model.fit(x, y)

# Koeffitsientlar
a = model.intercept_
b = model.coef_[0]

print("Model tenglamasi: y =", a, "+", b, "* x")

# Bashorat
y_pred = model.predict([[7]])
print("x = 7 uchun bashorat:", y_pred[0])
