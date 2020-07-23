import collections
import logging

from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.formats.base import BaseExperiment
from arekit.common.experiment.labeling import LabeledCollection
from arekit.common.labels.base import Label
from arekit.common.model.labeling.single import SingleLabelsHelper

from arekit.contrib.networks.context.configurations.base.base import DefaultNetworkConfig
from arekit.contrib.networks.sample import InputSample

from arekit.networks.input.readers.samples import NetworkInputSampleReader
from arekit.networks.input.rows_parser import ParsedSampleRow
from arekit.networks.feeding.bags.collection.base import BagsCollection

logger = logging.getLogger(__name__)


class HandledData(object):

    def __init__(self, labeled_collections, bags_collection):
        assert(isinstance(labeled_collections, dict))
        assert(isinstance(bags_collection, dict))
        self.__labeled_collections = labeled_collections
        self.__bags_collection = bags_collection

    @classmethod
    def initialize_from_experiment(cls, experiment, config, bags_collection_type):
        assert(isinstance(experiment, BaseExperiment))
        assert(isinstance(config, DefaultNetworkConfig))

        instance = cls(labeled_collections={}, bags_collection={})

        for data_type in experiment.DocumentOperations.iter_suppoted_data_types():
            labeled_sample_row_ids = instance.__init_for_data_type(experiment=experiment,
                                                                   config=config,
                                                                   data_type=data_type,
                                                                   bags_collection_type=bags_collection_type)

            if data_type != DataType.Train:
                continue

            labels_helper = SingleLabelsHelper(label_scaler=experiment.DataIO.LabelsScaler)
            norm, _ = instance.get_statistic(labeled_sample_row_ids=labeled_sample_row_ids,
                                             labels_helper=labels_helper)
            config.set_class_weights(norm)

        config.notify_initialization_completed()

    def __init_for_data_type(self, experiment, config, data_type, bags_collection_type):
        assert(isinstance(data_type, DataType))
        assert(issubclass(bags_collection_type, BagsCollection))

        samples_reader = NetworkInputSampleReader.from_tsv(filepath=None,
                                                           row_ids_provider=None)

        labeled_sample_row_ids = list(samples_reader.iter_labeled_sample_rows(
            label_scaler=experiment.DataIO.LabelsScaler))

        self.__labeled_collections[data_type] = LabeledCollection(labeled_sample_row_ids=labeled_sample_row_ids)

        self.__bags_collection[data_type] = bags_collection_type.from_formatted_samples(
            samples_reader=samples_reader,
            bag_size=config.BagSize,
            shuffle=True,
            create_empty_sample_func=lambda config: InputSample.create_empty(config),
            create_sample_func=lambda row: self.__create_input_sample(row=row, config=config))

        return labeled_sample_row_ids

    # region Properties

    @property
    def BagsCollections(self):
        return self.__bags_collection

    @property
    def SamplesLabelingCollection(self):
        return self.__labeled_collections

    # endregion

    @staticmethod
    def get_statistic(labeled_sample_row_ids, labels_helper):
        assert(isinstance(labeled_sample_row_ids, collections.Iterable))

        stat = [0] * labels_helper.get_classes_count()

        for _, label in labeled_sample_row_ids:
            assert(isinstance(label, Label))
            stat[labels_helper.label_to_uint(label)] += 1

        total = sum(stat)
        norm = [100.0 * value / total if total > 0 else 0 for value in stat]
        return norm, stat

    # region private methods

    def __create_input_sample(self, row, config):
        """
        Creates an input for Neural Network model
        """
        assert(isinstance(row, ParsedSampleRow))
        assert(isinstance(config, DefaultNetworkConfig))

        # TODO. Provide extra parameters.
        return InputSample.from_tsv_row(
            row_id=row.RowID,
            terms=row.Terms,
            subj_ind=row.SubjectIndex,
            obj_ind=row.ObjectIndex,
            words_vocab=None,
            config=config)

    @staticmethod
    def __create_collection(data_types, collection_by_dtype_func):
        assert(isinstance(data_types, collections.Iterable))
        assert(callable(collection_by_dtype_func))

        collection = {}
        for data_type in data_types:
            collection[data_type] = collection_by_dtype_func(data_type)

        return collection

    # endregion
