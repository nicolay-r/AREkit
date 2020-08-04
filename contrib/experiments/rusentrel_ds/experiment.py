import logging

from tqdm import tqdm

from arekit.common.experiment.formats.cv_based.experiment import CVBasedExperiment
from arekit.contrib.experiments.rusentrel.experiment import RuSentRelExperiment
from arekit.contrib.experiments.rusentrel_ds.documents import RuSentrelWithRuAttitudesDocumentOperations
from arekit.contrib.experiments.rusentrel_ds.opinions import RuSentrelWithRuAttitudesOpinionOperations
from arekit.processing.lemmatization.base import Stemmer
from arekit.source.ruattitudes.collection import RuAttitudesCollection
from arekit.source.ruattitudes.news.base import RuAttitudesNews

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class RuSentRelWithRuAttitudesExperiment(CVBasedExperiment):
    """
    IO for the experiment with distant supervision for sentiment attitude extraction task.
    Paper: https://www.aclweb.org/anthology/R19-1118/
    """

    def __init__(self, data_io, prepare_model_root, ra_instance=None):
        """
        ra_instance: dict
            precomputed ru_attitudes (in memory)
        """
        assert(isinstance(ra_instance, dict) or ra_instance is None)

        rusentrel_news_inds = RuSentRelExperiment.get_rusentrel_inds()

        ru_attitudes = ra_instance
        if ra_instance is None:
            ru_attitudes = RuSentRelWithRuAttitudesExperiment.read_ruattitudes_in_memory(data_io.Stemmer)

        doc_ops = RuSentrelWithRuAttitudesDocumentOperations(
            data_io=data_io,
            rusentrel_news_inds=rusentrel_news_inds)

        opin_ops = RuSentrelWithRuAttitudesOpinionOperations(
            data_io=data_io,
            annot_name_func=lambda: self.NeutralAnnotator.AnnotatorName,
            rusentrel_news_inds=rusentrel_news_inds)

        super(RuSentRelWithRuAttitudesExperiment, self).__init__(
            data_io=data_io,
            opin_ops=opin_ops,
            doc_ops=doc_ops,
            prepare_model_root=prepare_model_root)

        doc_ops.set_ru_attitudes(ru_attitudes)
        opin_ops.set_ru_attitudes(ru_attitudes)

    @staticmethod
    def read_ruattitudes_in_memory(stemmer, doc_ids_set=None):
        """
        Performs reading of ruattitude formatted documents and
        selection according to 'doc_ids_set' parameter.

        doc_ids_set: set or None
            ids of documents that should be selected.
            'None' corresponds to all the available doc_ids.
        """
        assert(isinstance(stemmer, Stemmer))
        assert(isinstance(doc_ids_set, set) or doc_ids_set is None)

        d = {}

        it = tqdm(iterable=RuAttitudesCollection.iter_news(stemmer=stemmer),
                  desc="Loading RuAttitudes collection in memory",
                  ncols=120)

        for news in it:
            assert(isinstance(news, RuAttitudesNews))

            if doc_ids_set is not None and news.ID not in doc_ids_set:
                continue

            d[news.ID] = news

        return d

