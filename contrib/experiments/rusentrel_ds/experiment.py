import logging
from arekit.common.experiment.formats.cv_based.experiment import CVBasedExperiment
from arekit.contrib.experiments.common import get_neutral_annotation_root
from arekit.contrib.experiments.ruattitudes.documents import RuAttitudesDocumentOperations
from arekit.contrib.experiments.ruattitudes.opinions import RuAttitudesOpinionOperations
from arekit.contrib.experiments.ruattitudes.utils import read_ruattitudes_in_memory
from arekit.contrib.experiments.rusentrel.documents import RuSentrelDocumentOperations
from arekit.contrib.experiments.rusentrel.experiment import RuSentRelExperiment
from arekit.contrib.experiments.rusentrel.folding_type import FoldingType
from arekit.contrib.experiments.rusentrel.opinions import RuSentrelOpinionOperations
from arekit.contrib.experiments.rusentrel.utils import folding_type_to_str
from arekit.contrib.experiments.rusentrel_ds.documents import RuSentrelWithRuAttitudesDocumentOperations
from arekit.contrib.experiments.rusentrel_ds.opinions import RuSentrelWithRuAttitudesOpinionOperations
from arekit.contrib.source.ruattitudes.io_utils import RuAttitudesVersions
from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions

logger = logging.getLogger(__name__)


class RuSentRelWithRuAttitudesExperiment(CVBasedExperiment):
    """
    IO for the experiment with distant supervision for sentiment attitude extraction task.
    Original Paper (RuAttitudes-1.0): https://www.aclweb.org/anthology/R19-1118/
    """

    def __init__(self, data_io, folding_type, ruattitudes_version, rusentrel_version, ra_instance=None):
        """
        ra_instance: dict
            precomputed ru_attitudes (in memory)
        """
        assert(isinstance(ruattitudes_version, RuAttitudesVersions))
        assert(isinstance(rusentrel_version, RuSentRelVersions))
        assert(isinstance(folding_type, FoldingType))
        assert(isinstance(ra_instance, dict) or ra_instance is None)

        self.__ruattitudes_version = ruattitudes_version
        self.__rusentrel_version = rusentrel_version

        super(RuSentRelWithRuAttitudesExperiment, self).__init__(data_io=data_io)

        rusentrel_news_inds = RuSentRelExperiment.get_rusentrel_inds()
        rusentrel_doc = RuSentrelDocumentOperations(data_io=data_io,
                                                    rusentrel_version=rusentrel_version,
                                                    folding_type=folding_type)
        ruattitudes_doc = RuAttitudesDocumentOperations(data_io=data_io)
        doc_ops = RuSentrelWithRuAttitudesDocumentOperations(rusentrel_doc=rusentrel_doc,
                                                             rusentrel_news_ids=rusentrel_news_inds,
                                                             ruattitudes_doc=ruattitudes_doc)

        neutral_root = get_neutral_annotation_root(self)

        rusentrel_op = RuSentrelOpinionOperations(data_io=data_io,
                                                  version=rusentrel_version,
                                                  neutral_root=neutral_root,
                                                  rusentrel_news_ids=rusentrel_news_inds)

        ruattitudes_op = RuAttitudesOpinionOperations(synonyms=data_io.SynonymsCollection,
                                                      neutral_root=neutral_root)

        opin_ops = RuSentrelWithRuAttitudesOpinionOperations(
            synonyms=data_io.SynonymsCollection,
            neutral_root=neutral_root,
            rusentrel_op=rusentrel_op,
            ruattitudes_op=ruattitudes_op)

        ru_attitudes = ra_instance
        if ra_instance is None:
            ru_attitudes = read_ruattitudes_in_memory(
                version=ruattitudes_version,
                used_doc_ids_set=rusentrel_news_inds)

        # Providing RuAttitudes instance into doc and opins instances.
        ruattitudes_doc.set_ru_attitudes(ru_attitudes)
        ruattitudes_op.set_ru_attitudes(ru_attitudes)

        self._set_opin_operations(opin_ops)
        self._set_doc_operations(doc_ops)

        # Composing experiment name
        self.__name = u"rsr-{rsr_version}-ra-{ra_version}-{folding_type}".format(
            rsr_version=self.__rusentrel_version.value,
            ra_version=self.__ruattitudes_version.value,
            folding_type=folding_type_to_str(folding_type))

    @property
    def Name(self):
        return self.__name
