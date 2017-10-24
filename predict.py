#!/usr/bin/python
# -*- coding: utf-8 -*-

from sklearn import svm

from core.output.vectors import CommonRelationVectorCollection

import io_utils


X_train = []
y_train = []
root = io_utils.train_root()
for i in io_utils.train_indices():
    vector_filepath = root + "art{}.vectors.txt".format(i)
    print vector_filepath
    collection = CommonRelationVectorCollection.from_file(vector_filepath)
    X_train += [item.vector for item in collection]
    y_train += [item.label for item in collection]

print len(X_train)

X_test = []
root = io_utils.test_root()
for i in io_utils.test_indices():
    vector_filepath = root + "art{}.vectors.txt".format(i)
    print vector_filepath
    collection = CommonRelationVectorCollection.from_file(vector_filepath)
    X_train += [item.vector for item in collection]

# fitting model
c = svm.SVC()
c.fit(X_train, y_train)

labels = c.predict(X_test)
