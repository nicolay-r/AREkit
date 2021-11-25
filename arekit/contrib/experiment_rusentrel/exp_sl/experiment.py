import logging

from arekit.common.experiment.api.base import BaseExperiment
from arekit.common.experiment.api.io_utils import BaseIOUtils
from arekit.common.folding.types import FoldingType
from arekit.contrib.experiment_rusentrel.common import entity_to_group_func
from arekit.contrib.experiment_rusentrel.exp_sl.documents import RuSentrelDocumentOperations
from arekit.contrib.experiment_rusentrel.exp_sl.folding import create_rusentrel_experiment_data_folding
from arekit.contrib.experiment_rusentrel.exp_sl.opinions import RuSentrelOpinionOperations
from arekit.contrib.experiment_rusentrel.synonyms.provider import RuSentRelSynonymsCollectionProvider
from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class RuSentRelExperiment(BaseExperiment):
    """
    Represents a cv-based experiment over RuSentRel collection,
    which supports train/test separation.
    utilized in papers:
        https://link.springer.com/chapter/10.1007/978-3-030-23584-0_10
        https://wwww.easychair.org/publications/download/pQrC
    """

    def __init__(self, exp_data, experiment_io_type, version, folding_type, extra_name_suffix,
                 do_log=True):
        assert(isinstance(version, RuSentRelVersions))
        assert(isinstance(folding_type, FoldingType))
        assert(issubclass(experiment_io_type, BaseIOUtils))
        assert(isinstance(do_log, bool))

        # Setup logging option.
        self._init_log_flag(do_log)

        self.__rsr_version = version
        self.__synonyms = None

        self.log_info("Init experiment io ...")
        experiment_io = experiment_io_type(self)

        self.log_info("Create opinion operations ... ")
        opin_ops = RuSentrelOpinionOperations(experiment_data=exp_data,
                                              version=version,
                                              experiment_io=experiment_io,
                                              get_synonyms_func=self._get_or_load_synonyms_collection)

        self.log_info("Create document operations ... ")
        folding = create_rusentrel_experiment_data_folding(folding_type=folding_type,
                                                           version=version,
                                                           docs_reader_func=lambda doc_id: doc_ops.get_doc(doc_id),
                                                           experiment_io=experiment_io)
        doc_ops = RuSentrelDocumentOperations(exp_data=exp_data,
                                              folding=folding,
                                              version=version,
                                              get_synonyms_func=self._get_or_load_synonyms_collection)

        exp_name = "rsr-{version}-{format}".format(version=version.value,
                                                   format=doc_ops.DataFolding.Name)

        super(RuSentRelExperiment, self).__init__(exp_data=exp_data,
                                                  experiment_io=experiment_io,
                                                  doc_ops=doc_ops,
                                                  opin_ops=opin_ops,
                                                  name=exp_name,
                                                  extra_name_suffix=extra_name_suffix)

    # region protected methods

    def _get_or_load_synonyms_collection(self):
        if self.__synonyms is None:
            self.log_info("Read synonyms collection ...")
            self.__synonyms = RuSentRelSynonymsCollectionProvider.load_collection(
                stemmer=self.DataIO.Stemmer,
                version=self.__rsr_version)

        return self.__synonyms

    # endregion

    def entity_to_group(self, entity):
        return entity_to_group_func(entity, synonyms=self.__synonyms)
