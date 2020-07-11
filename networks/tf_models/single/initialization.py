import collections
import logging

from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.formats.base import BaseExperiment
from arekit.common.experiment.labeling import LabeledCollection
from arekit.common.labels.base import Label
from arekit.common.model.labeling.single import SingleLabelsHelper

from arekit.contrib.networks.context.configurations.base.base import DefaultNetworkConfig
from arekit.contrib.networks.sample import InputSample

from arekit.networks.input.formatters.sample import NetworkSampleFormatter
from arekit.networks.input.readers.samples import NetworkInputSampleReader
from arekit.networks.input.rows_parser import ParsedSampleRow
from arekit.networks.training.bags.collection.single import SingleBagsCollection

logger = logging.getLogger(__name__)


class SingleInstanceModelExperimentInitializer(object):

    def __init__(self, experiment, config):
        assert(isinstance(experiment, BaseExperiment))
        assert(isinstance(config, DefaultNetworkConfig))

        self.__labeled_collections = {}
        self.__bags_collection = {}

        for data_type in experiment.DocumentOperations.iter_suppoted_data_types():
            self.__init_for_data_type(experiment=experiment,
                                      config=config,
                                      data_type=data_type)

        config.notify_initialization_completed()

    def __init_for_data_type(self, experiment, config, data_type):
        assert(isinstance(data_type, DataType))

        samples_reader = NetworkInputSampleReader.from_tsv(filepath=None,
                                                           row_ids_provider=None)

        # TODO. Incomplete.
        # TODO: Create this (as iteration over samples with the related labels).
        # TODO. Labels especially in case of Train data type.
        # TODO. Utilize rows_parser.py (in networks/input) in order to perform the latter.
        labeled_sample_row_ids = list()

        self.__labeled_collections[data_type] = LabeledCollection(labeled_sample_row_ids=labeled_sample_row_ids)

        self.__bags_collection[data_type] = self._BagCollectionType.from_formatted_samples(
            samples_reader=samples_reader,
            bag_size=config.BagSize,
            shuffle=True,
            create_empty_sample_func=self._create_empty_sample_func,
            create_sample_func=lambda row: self.__create_input_sample(row=row, config=config))

        if data_type != DataType.Train:
            return

        labels_helper = SingleLabelsHelper(label_scaler=experiment.DataIO.LabelsScaler)
        norm, _ = self.get_statistic(labeled_sample_row_ids=labeled_sample_row_ids,
                                     labels_helper=labels_helper)

        config.set_class_weights(norm)

    # region Properties

    @property
    def BagsCollections(self):
        return self.__bags_collection

    # TODO. Samples labeling collection
    @property
    def LabeledCollection(self):
        return self.__labeled_collections

    @property
    def _BagCollectionType(self):
        return SingleBagsCollection

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

    def _create_empty_sample_func(self, config):
        return None

    def __create_bags_collection(self, samples_reader, config):
        assert(isinstance(samples_reader, NetworkSampleFormatter))
        assert(isinstance(config, DefaultNetworkConfig))

        collection = self._BagCollectionType.from_formatted_samples(
            samples_reader=samples_reader,
            bag_size=config.BagSize,
            shuffle=True,
            create_empty_sample_func=self._create_empty_sample_func,
            create_sample_func=lambda row: self.__create_input_sample(row=row, config=config))

        return collection

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
