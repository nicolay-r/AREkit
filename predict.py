#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn import svm, neighbors, ensemble, model_selection, naive_bayes

from core.source.vectors import CommonRelationVectorCollection
from core.source.opinion import OpinionCollection, Opinion

import io_utils


def svc_param_selection(X_train, y_train):
    C = np.arange(0.2, 10, 0.2)
    param_grid = {'C': C}
    model = svm.SVC(C=1, kernel='linear', cache_size=1000)
    grid_search = model_selection.GridSearchCV(model, param_grid, cv=10)
    return grid_search


X_train = []
y_train = []
for i in io_utils.train_indices():
    vector_filepath = io_utils.train_root() + "art{}.vectors.txt".format(i)
    print vector_filepath
    collection = CommonRelationVectorCollection.from_file(vector_filepath)
    X_train += [item.vector for item in collection]
    y_train += [item.label for item in collection]

X_test = []
test_collections = []
for i in io_utils.test_indices():
    vector_filepath = io_utils.test_root() + "art{}.vectors.txt".format(i)
    print vector_filepath
    collection = CommonRelationVectorCollection.from_file(vector_filepath)
    test_collections.append(collection)
    X_test += [item.vector for item in collection]

CLASSIFIERS = {
    "svm": svm.SVC(C=1, kernel='linear', cache_size=1000),
    "knn": neighbors.KNeighborsClassifier(),
    "rf": ensemble.RandomForestClassifier(),
    "nb": naive_bayes.GaussianNB(),
    "svm-search": svc_param_selection
}

c = CLASSIFIERS['nb']
print c
c.fit(X_train, y_train)
labels = c.predict(X_test)

# fill
label_index = 0
test_opinions = []
for c in test_collections:
    opinions = OpinionCollection()
    for item in c:
        l = int(labels[label_index])
        item.set_label(l)
        opinions.add_opinion(
            Opinion(item.value_left,
                    item.value_right,
                    io_utils.int_to_sentiment(l),
                    unicode("current")))
        label_index += 1
    test_opinions.append(opinions)

print "train:"
print "-1 : {}".format(y_train.count(-1))
print "0 : {}".format(y_train.count(0))
print "1 : {}".format(y_train.count(1))

print "test:"
print "-1 : {}".format(np.count_nonzero(labels == -1))
print "0 : {}".format(np.count_nonzero(labels == 0))
print "1 : {}".format(np.count_nonzero(labels == 1))

# save
for i, n in enumerate(io_utils.test_indices()):
    opin_filepath = io_utils.test_root() + "art{}.opin.txt".format(n)
    test_opinions[i].save(opin_filepath)
