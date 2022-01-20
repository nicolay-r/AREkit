import logging

from arekit.common.experiment.api.base import BaseExperiment
from arekit.common.experiment.name_provider import ExperimentNameProvider
from arekit.contrib.experiment_rusentrel.common import create_text_parser
from arekit.contrib.experiment_rusentrel.exp_ds.documents import RuAttitudesDocumentOperations
from arekit.contrib.experiment_rusentrel.exp_ds.folding import create_ruattitudes_experiment_data_folding
from arekit.contrib.experiment_rusentrel.exp_ds.opinions import RuAttitudesOpinionOperations
from arekit.contrib.experiment_rusentrel.exp_ds.utils import read_ruattitudes_in_memory
from arekit.contrib.source.ruattitudes.entity.parser import RuAttitudesTextEntitiesParser
from arekit.contrib.source.ruattitudes.io_utils import RuAttitudesVersions

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class RuAttitudesExperiment(BaseExperiment):
    """ Application of distant supervision, especially for pretraining purposes.
        Suggested to utilize with a large RuAttitudes-format collections (v2.0-large).
    """

    def __init__(self, exp_data, experiment_io_type, version, load_docs, do_log):
        assert(isinstance(version, RuAttitudesVersions))
        assert(isinstance(load_docs, bool))
        assert(isinstance(do_log, bool))

        # Setup logging option.
        self._init_log_flag(do_log)

        self.__version = version
        self.__do_log = do_log

        self.log_info("Init experiment io ...")
        experiment_io = experiment_io_type(self)

        self.log_info("Loading RuAttitudes collection optionally [{version}] ...".format(version=version))
        ru_attitudes = read_ruattitudes_in_memory(version=version,
                                                  used_doc_ids_set=None,
                                                  keep_doc_ids_only=not load_docs)

        folding = create_ruattitudes_experiment_data_folding(
            doc_ids_to_fold=list(ru_attitudes.keys()))

        text_parser = create_text_parser(exp_data=exp_data,
                                         entities_parser=RuAttitudesTextEntitiesParser(),
                                         value_to_group_id_func=None)

        self.log_info("Create document operations ... ")
        doc_ops = RuAttitudesDocumentOperations(folding=folding,
                                                ru_attitudes=ru_attitudes,
                                                text_parser=text_parser)

        self.log_info("Create opinion operations ... ")
        opin_ops = RuAttitudesOpinionOperations(ru_attitudes=ru_attitudes)

        super(RuAttitudesExperiment, self).__init__(exp_data=exp_data,
                                                    experiment_io=experiment_io,
                                                    opin_ops=opin_ops,
                                                    doc_ops=doc_ops)

    def log_info(self, message, forced=False):
        assert (isinstance(message, str))
        if not self.__do_log and not forced:
            return
        logger.info(message)
