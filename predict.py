#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
from sklearn import svm, neighbors, ensemble, model_selection, naive_bayes

from core.source.synonyms import SynonymsCollection
from core.source.vectors import OpinionVectorCollection
from core.source.opinion import OpinionCollection, Opinion
from core.eval import Evaluator

import io_utils


# TODO. Move into eval class
def get_result_columns():
    return ['pos_prec',
            'neg_prec',
            'pos_recall',
            'neg_recall',
            'f1_pos',
            'f1_neg',
            'f1']


def show_classification_info(y_train, labels):
    print "train:"
    for c in [-1, 0, 1]:
        print "{}: {}".format(c, y_train.count(c))
    print "test:"
    for c in [-1, 0, 1]:
        print "{}: {}".format(c, np.count_nonzero(labels == c))


def create_results(test_collections, labels):
    assert(type(test_collections) == list)
    assert(type(labels) == np.ndarray)

    label_index = 0
    test_opinions = []

    synonyms = SynonymsCollection.from_file(io_utils.get_synonyms_filepath())
    for c in test_collections:
        opinions = OpinionCollection(None, synonyms)
        for item in c:
            l = int(labels[label_index])
            item.set_label(l)
            o = Opinion(item.value_left,
                        item.value_right,
                        io_utils.int_to_sentiment(l))

            if not opinions.has_opinion_by_synonyms(o) and l != 0:
                opinions.add_opinion(o)
            elif l != 0:
                print "Failed for o={}".format(o.to_unicode().encode('utf-8'))

            label_index += 1
        test_opinions.append(opinions)
    return test_opinions


def svc_param_selection(X_train, y_train):
    C = np.arange(0.2, 10, 0.2)
    param_grid = {'C': C}
    model = svm.SVC(C=1, kernel='linear', cache_size=1000)
    grid_search = model_selection.GridSearchCV(model, param_grid, cv=10)
    return grid_search


def get_method_root(method_name):
    return "{}/{}".format(io_utils.test_root(), method_name)


def create_train_data():
    X_train = []
    y_train = []
    for i in io_utils.train_indices():
        vector_filepath = io_utils.train_root() + "/art{}.vectors.txt".format(i)
        collection = OpinionVectorCollection.from_file(vector_filepath)
        X_train += [item.vector for item in collection]
        y_train += [item.label for item in collection]

    return X_train, y_train


def create_test_data():
    X_test = []
    test_collections = []
    for i in io_utils.test_indices():
        vector_filepath = io_utils.test_root() + "art{}.vectors.txt".format(i)
        collection = OpinionVectorCollection.from_file(vector_filepath)
        test_collections.append(collection)
        X_test += [item.vector for item in collection]

    return X_test, test_collections


print "Preparing Train Collection"
X_train, y_train = create_train_data()
print "Preparing Test Collection"
X_test, test_collections = create_test_data()

CLASSIFIERS = {
    "svm": svm.SVC(C=1, kernel='linear', cache_size=1000),
    "knn": neighbors.KNeighborsClassifier(),
    "rf": ensemble.RandomForestClassifier(),
    "nb": naive_bayes.GaussianNB(),
    # "svm-grid": svc_param_selection
}

# fit
for method_name in CLASSIFIERS.iterkeys():
    c = CLASSIFIERS[method_name]
    c.fit(X_train, y_train)
    labels = c.predict(X_test)
    print c
    show_classification_info(y_train, labels)
    test_opinions = create_results(test_collections, labels)

    if not os.path.exists(get_method_root(method_name)):
        os.mkdir(get_method_root(method_name))

    for i, n in enumerate(io_utils.test_indices()):
        opin_filepath = "{}/art{}.opin.txt".format(
           get_method_root(method_name), n)

        test_opinions[i].save(opin_filepath)

# evaluate
df = pd.DataFrame(columns=get_result_columns())

for method_name in CLASSIFIERS.iterkeys():
    e = Evaluator(
        io_utils.get_synonyms_filepath(),
        get_method_root(method_name),
        io_utils.get_etalon_root())
    r = e.evaluate(io_utils.test_indices())

    df.loc[method_name] = [r[c] for c in get_result_columns()]

df.T.to_csv("{}/_eval.txt".format(io_utils.test_root()))
