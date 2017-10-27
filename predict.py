#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn import svm

from core.source.vectors import CommonRelationVectorCollection
from core.source.opinion import OpinionCollection, Opinion

import io_utils

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

# fitting model
c = svm.SVC(C=10, kernel='linear')
c.fit(X_train, y_train)

# predict
labels = c.predict(X_test)
print labels

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

print "-1 : {}".format(np.count_nonzero(labels == -1))
print "0 : {}".format(np.count_nonzero(labels == 0))
print "1 : {}".format(np.count_nonzero(labels == 1))

# save
for i, n in enumerate(io_utils.test_indices()):
    opin_filepath = io_utils.test_root() + "art{}.opin.txt".format(n)
    test_opinions[i].save(opin_filepath)
