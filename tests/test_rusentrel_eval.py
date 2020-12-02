import unittest

from os import path
from os.path import dirname

from arekit.common.evaluation.evaluators.two_class import TwoClassEvaluator
from arekit.common.evaluation.results.base import BaseEvalResult
from arekit.common.evaluation.utils import OpinionCollectionsToCompareUtils
from arekit.common.opinions.collection import OpinionCollection
from arekit.common.utils import progress_bar_iter
from arekit.contrib.source.rusentrel.io_utils import RuSentRelIOUtils, RuSentRelVersions
from arekit.contrib.source.rusentrel.labels_fmt import RuSentRelLabelsFormatter
from arekit.contrib.source.rusentrel.opinions.collection import RuSentRelOpinionCollection
from arekit.contrib.source.rusentrel.opinions.formatter import RuSentRelOpinionCollectionFormatter
from arekit.contrib.source.rusentrel.synonyms import RuSentRelSynonymsCollection
from arekit.contrib.source.zip_utils import ZipArchiveUtils
from arekit.processing.lemmatization.mystem import MystemWrapper


class CNNResultsRuSentRelIOUtils(ZipArchiveUtils):

    @staticmethod
    def get_archive_filepath(version):
        return path.join(dirname(__file__), u"data/att-cnn-fixed-20e-f1-41.zip".format(version=version))


class TestRuSentRelEvaluation(unittest.TestCase):

    __version = RuSentRelVersions.V11

    def __iter_etalon_opinions(self, doc_id):
        return RuSentRelOpinionCollection.iter_opinions_from_doc(doc_id=doc_id)

    def __iter_model_result_opinions(self, doc_id):
        return CNNResultsRuSentRelIOUtils.iter_from_zip(
            inner_path=path.join(u"{}.opin.txt".format(doc_id)),
            process_func=lambda input_file: RuSentRelOpinionCollectionFormatter._iter_opinions_from_file(
                input_file=input_file,
                labels_formatter=RuSentRelLabelsFormatter()),
            version=self.__version)

    def test_cnn_results_eval(self):

        # Initializing stemmer.
        stemmer = MystemWrapper()
        synonyms = RuSentRelSynonymsCollection.load_collection(stemmer)

        # Iter cmp opinions.
        cmp_pairs_iter = OpinionCollectionsToCompareUtils.iter_comparable_collections(
            doc_ids=RuSentRelIOUtils.iter_test_indices(self.__version),
            read_etalon_collection_func=lambda doc_id: OpinionCollection.init_as_custom(
                self.__iter_etalon_opinions(doc_id),
                synonyms),
            read_result_collection_func=lambda doc_id: OpinionCollection.init_as_custom(
                self.__iter_model_result_opinions(doc_id),
                synonyms))

        # getting evaluator.
        evaluator = TwoClassEvaluator()

        # evaluate every document.
        logged_cmp_pairs_it = progress_bar_iter(cmp_pairs_iter, desc=u"Evaluate", unit=u'pairs')
        result = evaluator.evaluate(cmp_pairs=logged_cmp_pairs_it)
        assert(isinstance(result, BaseEvalResult))

        # calculate results.
        result.calculate()

        print result.get_result_as_str()


if __name__ == '__main__':
    unittest.main()
