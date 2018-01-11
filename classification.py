from eval import Evaluator
from labels import Label, NeutralLabel
from statistic import MethodStatistic
from source.synonyms import SynonymsCollection
from source.opinion import OpinionCollection, Opinion
from core.source.vectors import OpinionVectorCollection

import numpy as np
import pandas as pd
from sklearn import svm, neighbors, ensemble, model_selection, naive_bayes


def create_train_data(vectors_filepath_list):
    assert(type(vectors_filepath_list) == list)
    X_train = []
    y_train = []
    for vector_filepath in vectors_filepath_list:
        collection = OpinionVectorCollection.from_file(vector_filepath)
        X_train += [item.vector for item in collection]
        y_train += [item.label.to_int() for item in collection]

    return X_train, y_train


def create_test_data(vectors_filepath_list):
    assert(type(vectors_filepath_list) == list)
    X_test = []
    test_collections = []
    for vector_filepath in vectors_filepath_list:
        collection = OpinionVectorCollection.from_file(vector_filepath)
        test_collections.append(collection)
        X_test += [item.vector for item in collection]

    return X_test, test_collections


def evaluate(estimator, method_name, files_to_compare_list,
             method_root_filepath, synonyms_filepath):

    df = MethodStatistic.get_method_statistic(
            files_to_compare_list,
            synonyms_filepath)

    df.index.name = method_name

    e = Evaluator(synonyms_filepath, method_root_filepath)
    r = e.evaluate(files_to_compare_list)

    for c in Evaluator.get_result_columns():
        df.loc[c] = None
        df.loc[c][0] = r[c]

    settings = get_estimator_settings(estimator)
    if settings is not None:
        for s_name in settings.iterkeys():
            r_name = 's_{}'.format(s_name)
            df.loc[r_name] = None
            df.loc[r_name][0] = settings[s_name]

    return df


def get_estimator_settings(method):
    if isinstance(method, svm.SVC):
        return {
            'C': method.C,
            'class_weight': method.class_weight,
            'kernel': method.kernel
        }
    elif isinstance(method, ensemble.RandomForestClassifier):
        return {
            'n_estimators': method.n_estimators,
            'max_depth': method.max_depth,
            'min_samples_leaf': method.min_samples_leaf,
            'class_weight': method.class_weight
        }
    elif isinstance(method, naive_bayes.BernoulliNB):
        return {
            'binarize': method.binarize,
            'alpha': method.alpha
        }
    elif isinstance(method, naive_bayes.GaussianNB):
        return {}
    elif isinstance(method, neighbors.KNeighborsClassifier):
        return {
            'n_neighbors': method.n_neighbors
        }
    elif isinstance(method, model_selection.GridSearchCV):
        result = method.best_params_
        result['cv'] = method.cv
        result['scoring'] = method.scoring
        return result

    return None


def fit_and_predict(method_path, estimator, X_train, y_train, X_test,
                    test_collections, synonyms_filepath):
    """ Fitting the approptiate method and then predicting
    """
    estimator.fit(X_train, y_train)

    labels = estimator.predict(X_test)

    print method_path
    print "train: [{}]".format(" ".join(
            [str(y_train.count(c)) for c in [-1, 0, 1]]
        ))
    print "test: [{}]".format(" ".join(
            [str(np.count_nonzero(labels == c)) for c in [-1, 0, 1]]
        ))

    return create_test_opinions(test_collections, labels, synonyms_filepath)


def create_test_opinions(test_collections, labels, synonyms_filepath):
    assert(type(test_collections) == list)
    assert(type(labels) == np.ndarray)

    label_index = 0
    opinion_collection_list = []

    synonyms = SynonymsCollection.from_file(synonyms_filepath)
    for c in test_collections:
        opinions = OpinionCollection(None, synonyms)
        for opinion_vector in c:
            l = Label.from_int(int(labels[label_index]))
            opinion_vector.set_label(l)
            o = Opinion(opinion_vector.value_left,
                        opinion_vector.value_right,
                        opinion_vector.label)

            if not opinions.has_opinion_by_synonyms(o) and not isinstance(l, NeutralLabel):
                opinions.add_opinion(o)
            elif not isinstance(l, NeutralLabel):
                print "Failed for o={}".format(o.to_unicode().encode('utf-8'))

            label_index += 1
        opinion_collection_list.append(opinions)
    return opinion_collection_list


def filter_features(model_name, model, X_train, X_test):
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    print X_test.shape
    X_train_new = model.transform(X_train)
    X_test_new = model.transform(X_test)
    print "Train transform: {} -> {}".format(X_train.shape, X_train_new.shape)
    print "Test transform: {} -> {}".format(X_test.shape, X_test_new.shape)
    return (X_train_new, X_test_new)


def filter_features_by_mask(X_train, X_test, mask):
    """
        mask : list
            list of boolean values
    """
    assert(type(mask) == list)
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    print X_train.shape
    print X_test.shape
    return (X_train[:, mask], X_test[:, mask])


def create_model_feature_importances(model, method_name, support, feature_names):
    """
        support: list
            list of boolean values, where 'true' corresponds to selected
            features.
    """
    df = pd.DataFrame(columns=['support', 'coef', 'importance'])

    for i, s in enumerate(support):
        row_id = feature_names[i]
        df.loc[row_id] = None
        if s:
            df.loc[row_id]['support'] = '+'

    if hasattr(model, 'coef_'):
        for i, coef in enumerate(model.coef_):
            row_id = feature_names[i]
            df.loc[row_id] = None
            df.loc[row_id]['coef'] = coef[0]

    if hasattr(model, 'feature_importances_'):
        for i, importance in enumerate(model.feature_importances_):
            row_id = feature_names[i]
            if row_id not in df.index:
                df.loc[row_id] = None
            df.loc[row_id]['importance'] = importance

    return df


def plot_feature_importances(filepath):
    offset_width = 0.2
    row_width = 0.5
    df = pd.read_csv(filepath, index_col=0)
    df = df.sort_values(['coef', 'importance'], ascending=[True, True])
    ax = df.plot(kind='barh',
                 width=row_width,
                 figsize=(15, offset_width * len(df)))
    fig = ax.get_figure()
    fig.tight_layout()
    return fig
