import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from src.linreg import LinearRegressionGD

X, y = load_diabetes(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
Xtr = (Xtr - mu) / sd
Xte = (Xte - mu) / sd

mine = LinearRegressionGD(lr=0.1, n_iter=500).fit(Xtr, ytr)
ref = LinearRegression().fit(Xtr, ytr)

print("My model   test MSE:", round(mean_squared_error(yte, mine.predict(Xte)), 2))
print("scikit-learn test MSE:", round(mean_squared_error(yte, ref.predict(Xte)), 2))

plt.plot(mine.losses)
plt.xlabel("Iteration")
plt.ylabel("Training MSE")
plt.title("Gradient descent loss curve")
plt.savefig("loss_curve.png", dpi=150)
print("Saved loss_curve.png")