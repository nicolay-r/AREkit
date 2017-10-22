#!/usr/bin/python
# -*- coding: utf-8 -*-

from sklearn import svm

from core.source.vectors import NewsVectorizedRelations


root = "data/Texts/"

# read train dataset

X_train = []
y_train = []
for i in range(1, 46):
    vector_filepath = root + "art{}.vectors.txt".format(i)
    vectors = NewsVectorizedRelations.from_file(vector_filepath, labeled=True)
    X_train += vectors.X # refactor as concat
    y_train += vectors.labels

X_test = []
for i in range(46, 76):
    vector_filepath = root + "art{}.vectors.txt".format(i)
    vectors = NewsVectorizedRelations.from_file(vector_filepat)
    X_test += vectors.X # refactor as concat


# fitting model
c = svm.SVC()
c.fit(X_train, y_train)

labels = c.predict(X_test)
