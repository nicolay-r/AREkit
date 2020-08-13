import logging

from arekit.common.experiment.formats.cv_based.experiment import CVBasedExperiment
from arekit.common.utils import progress_bar_iter
from arekit.contrib.experiments.rusentrel.experiment import RuSentRelExperiment
from arekit.contrib.experiments.rusentrel_ds.documents import RuSentrelWithRuAttitudesDocumentOperations
from arekit.contrib.experiments.rusentrel_ds.opinions import RuSentrelWithRuAttitudesOpinionOperations
from arekit.contrib.source.ruattitudes.collection import RuAttitudesCollection
from arekit.contrib.source.ruattitudes.io_utils import RuAttitudesVersions
from arekit.contrib.source.ruattitudes.news.base import RuAttitudesNews
from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions

logger = logging.getLogger(__name__)


class RuSentRelWithRuAttitudesExperiment(CVBasedExperiment):
    """
    IO for the experiment with distant supervision for sentiment attitude extraction task.
    Paper: https://www.aclweb.org/anthology/R19-1118/
    """

    def __init__(self, data_io, prepare_model_root, version, rusentrel_version, ra_instance=None):
        """
        ra_instance: dict
            precomputed ru_attitudes (in memory)
        """
        assert(isinstance(version, RuAttitudesVersions))
        assert(isinstance(rusentrel_version, RuSentRelVersions))
        assert(isinstance(ra_instance, dict) or ra_instance is None)

        self.__version = version
        self.__rusentrel_version = rusentrel_version

        super(RuSentRelWithRuAttitudesExperiment, self).__init__(
            data_io=data_io,
            prepare_model_root=prepare_model_root)

        rusentrel_news_inds = RuSentRelExperiment.get_rusentrel_inds()

        doc_ops = RuSentrelWithRuAttitudesDocumentOperations(
            data_io=data_io,
            version=version,
            rusentrel_news_inds=rusentrel_news_inds)

        opin_ops = RuSentrelWithRuAttitudesOpinionOperations(
            data_io=data_io,
            neutral_annot_name=self.get_annot_name(),
            experiments_name=self.Name,
            rusentrel_news_inds=rusentrel_news_inds,
            rusetrel_version=rusentrel_version)

        ru_attitudes = ra_instance
        if ra_instance is None:
            ru_attitudes = RuSentRelWithRuAttitudesExperiment.read_ruattitudes_in_memory(version)

        doc_ops.set_ru_attitudes(ru_attitudes)
        opin_ops.set_ru_attitudes(ru_attitudes)

        self._set_opin_operations(opin_ops)
        self._set_doc_operations(doc_ops)

    @property
    def Name(self):
        return u"rsr-{rsr_version}-ra-{ra_version}".format(
            rsr_version=self.__rusentrel_version.value,
            ra_version=self.__version.value)

    @staticmethod
    def read_ruattitudes_in_memory(version, doc_ids_set=None):
        """
        Performs reading of ruattitude formatted documents and
        selection according to 'doc_ids_set' parameter.

        doc_ids_set: set or None
            ids of documents that should be selected.
            'None' corresponds to all the available doc_ids.
        """
        assert(isinstance(version, RuAttitudesVersions))
        assert(isinstance(doc_ids_set, set) or doc_ids_set is None)

        id_offset = max(doc_ids_set) + 1 if doc_ids_set is not None else 0

        d = {}

        news_it = progress_bar_iter(
            iterable=RuAttitudesCollection.iter_news(
                version=version,
                get_news_index_func=lambda: id_offset + len(d)),
            desc=u"Loading RuAttitudes collection in memory, please wait ...",
            unit=u'docs')

        for news in news_it:
            assert(isinstance(news, RuAttitudesNews))

            if doc_ids_set is not None and news.ID not in doc_ids_set:
                continue

            d[news.ID] = news

        return d

