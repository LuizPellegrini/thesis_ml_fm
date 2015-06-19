import matplotlib.pyplot as pyplot

from sklearn import datasets
from sklearn import svm

digits = datasets.load_digits()

clf = svm.SVC(gamma=0.001, C=100)

print(len(digits.data))

x,y = digits.data[:-1], digits.target[:-1]
