from sklearn import svm


# add file reading
X_train = [[0, 0], [1, 1]]
X_test = [[2, 2]]

# read results
y_train = [0, 1]

c = svm.SVC()

c.fit(X_train, y_train)
labels = c.predict(X_test)
