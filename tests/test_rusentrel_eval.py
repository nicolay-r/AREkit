import unittest
import pandas as pd

from os import path
from os.path import dirname

from enum import Enum

from arekit.common.evaluation.evaluators.two_class import TwoClassEvaluator
from arekit.common.evaluation.results.two_class import TwoClassEvalResult
from arekit.common.evaluation.utils import OpinionCollectionsToCompareUtils
from arekit.common.opinions.collection import OpinionCollection
from arekit.common.utils import progress_bar_iter
from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions
from arekit.contrib.source.rusentrel.labels_fmt import RuSentRelLabelsFormatter
from arekit.contrib.source.rusentrel.opinions.collection import RuSentRelOpinionCollection
from arekit.contrib.source.rusentrel.opinions.formatter import RuSentRelOpinionCollectionFormatter
from arekit.contrib.source.rusentrel.synonyms import RuSentRelSynonymsCollection
from arekit.contrib.source.zip_utils import ZipArchiveUtils
from arekit.processing.lemmatization.mystem import MystemWrapper


class ResultVersions(Enum):

    # Results with a fixed document separation.
    AttCNNFixed = u"att-cnn-fixed-20e-f1-41.zip"
    AttPCNNFixed = u"att-pcnn-fixed-26e-f1-41.zip"

    # Results with cv split.
    AttPCNNCV3e40i0 = u"cv3_att-pcnn_e40_i0.zip"
    AttPCNNCV3e40i1 = u"cv3_att-pcnn_e40_i1.zip"
    AttPCNNCV3e40i2 = u"cv3_att-pcnn_e40_i2.zip"


class ZippedResultsIOUtils(ZipArchiveUtils):

    @staticmethod
    def get_archive_filepath(result_version):
        return path.join(dirname(__file__), u"data/{version}".format(version=result_version))

    @staticmethod
    def iter_doc_ids(result_version):
        for f_name in ZippedResultsIOUtils.iter_filenames_from_zip(result_version):
            doc_id_str = f_name.split('.')[0]
            yield int(doc_id_str)

    @staticmethod
    def iter_doc_opinions(doc_id, result_version):
        return ZippedResultsIOUtils.iter_from_zip(
            inner_path=path.join(u"{}.opin.txt".format(doc_id)),
            process_func=lambda input_file: RuSentRelOpinionCollectionFormatter._iter_opinions_from_file(
                input_file=input_file,
                labels_formatter=RuSentRelLabelsFormatter()),
            version=result_version)


class TestRuSentRelEvaluation(unittest.TestCase):

    __rusentrel_version = RuSentRelVersions.V11

    def __test_core(self, res_version):
        assert(isinstance(res_version, ResultVersions))

        # Initializing stemmer.
        stemmer = MystemWrapper()
        synonyms = RuSentRelSynonymsCollection.load_collection(stemmer)

        # Iter cmp opinions.
        cmp_pairs_iter = OpinionCollectionsToCompareUtils.iter_comparable_collections(
            doc_ids=ZippedResultsIOUtils.iter_doc_ids(res_version),
            read_etalon_collection_func=lambda doc_id: OpinionCollection.init_as_custom(
                RuSentRelOpinionCollection.iter_opinions_from_doc(doc_id=doc_id),
                synonyms),
            read_result_collection_func=lambda doc_id: OpinionCollection.init_as_custom(
                ZippedResultsIOUtils.iter_doc_opinions(doc_id=doc_id,
                                                       result_version=res_version),
                synonyms))

        # getting evaluator.
        evaluator = TwoClassEvaluator()

        # evaluate every document.
        logged_cmp_pairs_it = progress_bar_iter(cmp_pairs_iter, desc=u"Evaluate", unit=u'pairs')
        result = evaluator.evaluate(cmp_pairs=logged_cmp_pairs_it)
        assert(isinstance(result, TwoClassEvalResult))

        # calculate results.
        result.calculate()

        # logging all the result information.
        for doc_id, doc_info in result.iter_document_results():
            print u"{}:\t{}".format(doc_id, doc_info)
        print "------------------------"
        print result.get_result_as_str()
        print "------------------------"

        # Display cmp tables.
        # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        #     for doc_id, df_cmp_table in result.iter_dataframe_cmp_tables():
        #         print u"{}:\t{}\n".format(doc_id, df_cmp_table)
        # print "------------------------"

    def test_ann_cnn(self):
        self.__test_core(ResultVersions.AttCNNFixed)

    def test_ann_pcnn(self):
        self.__test_core(ResultVersions.AttPCNNFixed)

    def test_ann_pcnn_cv(self):
        self.__test_core(ResultVersions.AttPCNNCV3e40i0)
        # self.__test_core(ResultVersions.AttPCNNCV3e40i1)
        # self.__test_core(ResultVersions.AttPCNNCV3e40i2)


if __name__ == '__main__':
    unittest.main()
