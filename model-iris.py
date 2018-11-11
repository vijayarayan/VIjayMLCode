from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data = load_iris()

print("data %s",data)

X, y= load_iris(return_X_y=True)