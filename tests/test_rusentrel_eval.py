import unittest
import pandas as pd

from os import path
from os.path import dirname

from enum import Enum

from arekit.common.evaluation.evaluators.two_class import TwoClassEvaluator
from arekit.common.evaluation.results.two_class import TwoClassEvalResult
from arekit.common.evaluation.utils import OpinionCollectionsToCompareUtils
from arekit.common.opinions.collection import OpinionCollection
from arekit.common.synonyms import SynonymsCollection
from arekit.common.utils import progress_bar_iter
from arekit.contrib.source.ruattitudes.io_utils import RuAttitudesVersions
from arekit.contrib.source.ruattitudes.synonyms_helper import RuAttitudesSynonymsCollectionHelper
from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions
from arekit.contrib.source.rusentrel.labels_fmt import RuSentRelLabelsFormatter
from arekit.contrib.source.rusentrel.opinions.collection import RuSentRelOpinionCollection
from arekit.contrib.source.rusentrel.opinions.formatter import RuSentRelOpinionCollectionFormatter
from arekit.contrib.source.rusentrel.synonyms import StemmerBasedSynonymCollection
from arekit.contrib.source.rusentrel.synonyms_helper import RuSentRelSynonymsCollectionHelper
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

    # Distant Supervision + Supervised Learning.
    # results check.
    DSAttCNNFixedE40 = u"ds_att-cnn-fixed_e40.zip"

    # Could not reproduce F1=0.40, only f1=0.37 in 0.20.5
    PCNNLrecFixedE29 = u"pcnn-lrec-ds-e27-fixed.zip"


# Expected F1-values for every result.
f1_rusentrel_v11_results = {
    ResultVersions.DSAttCNNFixedE40: 0.40848820278013587,
    ResultVersions.AttPCNNCV3e40i0: 0.31908734912456854,
    ResultVersions.AttPCNNCV3e40i1: 0.29308682656891705,
    ResultVersions.AttPCNNCV3e40i2: 0.27847993499755,
    ResultVersions.AttCNNFixed: 0.2992231753125483,
    ResultVersions.AttPCNNFixed: 0.3476705309623523,
    ResultVersions.PCNNLrecFixedE29: 0.3710003588082132
}


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

    __display_cmp_table = False
    __rusentrel_version = RuSentRelVersions.V11

    @staticmethod
    def __create_stemmer():
        return MystemWrapper()

    def __iter_synonyms_group_lists(self):
        for group in RuSentRelSynonymsCollectionHelper.iter_groups(self.__rusentrel_version):
            yield group
        for group in RuAttitudesSynonymsCollectionHelper.iter_groups(RuAttitudesVersions.V20LargeNeut):
            yield group

    def __is_equal_results(self, v1, v2):
        self.assert_(abs(v1 - v2) < 1e-10)

    def __test_core(self, res_version, synonyms=None):
        assert(isinstance(res_version, ResultVersions))
        assert(isinstance(synonyms, SynonymsCollection) or synonyms is None)

        # Initializing synonyms collection.
        if synonyms is None:
            # This is a default collection which we used
            # to provide the results in `f1_rusentrel_v11_results`.
            stemmer = self.__create_stemmer()
            actual_synonyms = RuSentRelSynonymsCollectionHelper.load_collection(stemmer)
        else:
            actual_synonyms = synonyms

        # Iter cmp opinions.
        cmp_pairs_iter = OpinionCollectionsToCompareUtils.iter_comparable_collections(
            doc_ids=ZippedResultsIOUtils.iter_doc_ids(res_version),
            read_etalon_collection_func=lambda doc_id: OpinionCollection(
                opinions=RuSentRelOpinionCollection.iter_opinions_from_doc(doc_id=doc_id),
                synonyms=actual_synonyms,
                error_on_duplicates=False,
                error_on_synonym_end_missed=True),
            read_result_collection_func=lambda doc_id: OpinionCollection(
                opinions=ZippedResultsIOUtils.iter_doc_opinions(doc_id=doc_id, result_version=res_version),
                synonyms=actual_synonyms,
                error_on_duplicates=False,
                error_on_synonym_end_missed=False))

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

        # Display cmp tables (optionally).
        if self.__display_cmp_table:
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                for doc_id, df_cmp_table in result.iter_dataframe_cmp_tables():
                    print u"{}:\t{}\n".format(doc_id, df_cmp_table)
            print "------------------------"
        self.__is_equal_results(v1=result.get_result_by_metric(TwoClassEvalResult.C_F1),
                                v2=f1_rusentrel_v11_results[res_version])

    def test_ann_cnn(self):
        self.__test_core(ResultVersions.AttCNNFixed)

    def test_ann_pcnn(self):
        self.__test_core(ResultVersions.AttPCNNFixed)

    def test_ann_cnn_ds(self):
        self.__test_core(ResultVersions.DSAttCNNFixedE40)

    def test_ann_pcnn_cv(self):
        self.__test_core(ResultVersions.AttPCNNCV3e40i0)
        self.__test_core(ResultVersions.AttPCNNCV3e40i1)
        self.__test_core(ResultVersions.AttPCNNCV3e40i2)

    def test_pcnn_lrec(self):
        self.__test_core(ResultVersions.PCNNLrecFixedE29)

    def test_rsr_ra_20_merged_collection(self):

        synonyms = StemmerBasedSynonymCollection(
            iter_group_values_lists=self.__iter_synonyms_group_lists(),
            stemmer=self.__create_stemmer(),
            is_read_only=True,
            debug=False)

        self.__test_core(ResultVersions.PCNNLrecFixedE29,
                         synonyms=synonyms)


if __name__ == '__main__':
    unittest.main()
