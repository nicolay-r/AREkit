#!/usr/bin/python
# -*- coding: utf-8 -*-

from sklearn import svm

from core.source.vectors import NewsVectorizedRelations

import io_utils


X_train = []
y_train = []
root = io_utils.train_root()
for i in io_utils.train_indices():
    vector_filepath = root + "art{}.vectors.txt".format(i)
    vectors = NewsVectorizedRelations.from_file(vector_filepath, labeled=True)
    X_train += vectors.X  # refactor as concat
    y_train += vectors.labels

print len(X_train)

X_test = []
root = io_utils.test_root()
for i in io_utils.test_indices():
    vector_filepath = root + "art{}.vectors.txt".format(i)
    vectors = NewsVectorizedRelations.from_file(vector_filepath)
    X_test += vectors.X  # refactor as concat

# fitting model
c = svm.SVC()
c.fit(X_train, y_train)

labels = c.predict(X_test)
