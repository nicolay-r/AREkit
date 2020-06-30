import os
import logging
from arekit.common.experiment.data_io import DataIO
from arekit.common.linked.text_opinions.collection import LinkedTextOpinionCollection
from arekit.common.model.labeling.base import LabelsHelper


logger = logging.getLogger(__name__)


def write_log(data_io, log_names, log_values):
    assert(isinstance(data_io, DataIO))
    assert(isinstance(log_names, list))
    assert(isinstance(log_values, list))
    assert(len(log_names) == len(log_values))

    log_path = os.path.join(data_io.get_model_root(), u"log.txt")

    with open(log_path, 'w') as f:
        for index, log_value in enumerate(log_values):
            f.write("{}: {}\n".format(log_names[index], log_value))


def debug_labels_statistic(collection, name, labels_helper, stat_func):
    """
    stat_func: (collection, labels_helper) -> norm, stat
    """
    assert(isinstance(name, unicode))
    assert(isinstance(collection, LinkedTextOpinionCollection))
    assert(isinstance(labels_helper, LabelsHelper))
    assert(callable(stat_func))

    norm, stat = stat_func(collection, labels_helper)
    total = len(collection)

    logger.info("Extracted relation collection: {}".format(name))
    logger.info("\tTotal: {}".format(total))

    for i, value in enumerate(norm):
        label = labels_helper.label_from_uint(i)
        logger.info("\t{}: {:.2f}%\t({} relations)".format(label.to_class_str(), value, stat[i]))


def debug_unique_relations_statistic(name, collection):
    assert(isinstance(name, unicode))
    assert(isinstance(collection, LinkedTextOpinionCollection))

    statistic = __get_group_statistic(collection)
    total = sum(list(statistic.itervalues()))

    logger.info("Unique linked_text_opinions statistic: {}".format(name))
    logger.info("\tTotal: {}".format(total))

    for key, value in sorted(statistic.iteritems()):
        logger.info("\t{} -- {} ({:.2f}%)".format(key, value, 100.0 * value / total))
        total += value


def __get_group_statistic(collection):
    assert(isinstance(collection, LinkedTextOpinionCollection))

    stat = {}
    for linked_wrap in collection.iter_wrapped_linked_text_opinions():
        key = len(linked_wrap)
        stat[key] = 1 if key not in stat else stat[key] + 1

    return stat
