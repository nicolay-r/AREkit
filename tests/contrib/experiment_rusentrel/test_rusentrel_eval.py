import unittest
import pandas as pd

from os import path
from os.path import dirname

from enum import Enum

from arekit.common.evaluation.evaluators.cmp_table import DocumentCompareTable
from arekit.common.evaluation.evaluators.modes import EvaluationModes
from arekit.common.evaluation.utils import OpinionCollectionsToCompareUtils
from arekit.common.opinions.collection import OpinionCollection
from arekit.common.synonyms import SynonymsCollection
from arekit.common.utils import progress_bar_iter
from arekit.contrib.experiment_rusentrel.evaluation.evaluators.two_class import TwoClassEvaluator
from arekit.contrib.experiment_rusentrel.evaluation.results.two_class import TwoClassEvalResult
from arekit.contrib.experiment_rusentrel.labels.formatters.rusentiframes import ExperimentRuSentiFramesLabelsFormatter
from arekit.contrib.experiment_rusentrel.synonyms.provider import RuSentRelSynonymsCollectionProvider
from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions
from arekit.contrib.source.rusentrel.opinions.collection import RuSentRelOpinionCollection
from arekit.contrib.source.rusentrel.opinions.provider import RuSentRelOpinionCollectionProvider
from arekit.contrib.source.zip_utils import ZipArchiveUtils
from arekit.processing.lemmatization.mystem import MystemWrapper


class ResultVersions(Enum):

    # Results with a fixed document separation.
    AttCNNFixed = "att-cnn-fixed-20e-f1-41.zip"
    AttPCNNFixed = "att-pcnn-fixed-26e-f1-41.zip"

    # Results with cv split.
    AttPCNNCV3e40i0 = "cv3_att-pcnn_e40_i0.zip"
    AttPCNNCV3e40i1 = "cv3_att-pcnn_e40_i1.zip"
    AttPCNNCV3e40i2 = "cv3_att-pcnn_e40_i2.zip"

    # Distant Supervision + Supervised Learning.
    # results check.
    DSAttCNNFixedE40 = "ds_att-cnn-fixed_e40.zip"

    # Could not reproduce F1=0.40, only f1=0.37 in 0.20.5
    PCNNLrecFixedE29 = "pcnn-lrec-ds-e27-fixed.zip"

    CNNRsrRa20LargeNeut = "rsr-ra-20-large-neut-cnn.zip"

    SelfTestClassification = "self-rusentrel-11.zip"


# Expected F1-values for every result.
f1_rusentrel_v11_results = {
    # Extraction.
    ResultVersions.DSAttCNNFixedE40: 0.408487389179856,
    ResultVersions.AttPCNNCV3e40i0: 0.31908777332585647,
    ResultVersions.AttPCNNCV3e40i1: 0.2930870695297444,
    ResultVersions.AttPCNNCV3e40i2: 0.2784798325065801,
    ResultVersions.AttCNNFixed: 0.299223261072169,
    ResultVersions.AttPCNNFixed: 0.3476699877690299,
    ResultVersions.PCNNLrecFixedE29: 0.37100110911501544,
    ResultVersions.CNNRsrRa20LargeNeut: 0.4166391654898648,
    # Classification.
    ResultVersions.SelfTestClassification: 1.0
}


class ZippedResultsIOUtils(ZipArchiveUtils):

    @staticmethod
    def get_archive_filepath(result_version):
        return path.join(dirname(__file__), "data/{version}".format(version=result_version))

    @staticmethod
    def iter_doc_ids(result_version):
        for f_name in ZippedResultsIOUtils.iter_filenames_from_zip(result_version):
            doc_id_str = f_name.split('.')[0]
            yield int(doc_id_str)

    @staticmethod
    def iter_doc_opinions(doc_id, result_version, labels_formatter):
        return ZippedResultsIOUtils.iter_from_zip(
            inner_path=path.join("{}.opin.txt".format(doc_id)),
            process_func=lambda input_file: RuSentRelOpinionCollectionProvider._iter_opinions_from_file(
                input_file=input_file,
                labels_formatter=labels_formatter,
                error_on_non_supported=True),
            version=result_version)


class TestRuSentRelEvaluation(unittest.TestCase):

    __display_cmp_table = False
    __rusentrel_version = RuSentRelVersions.V11

    @staticmethod
    def __create_stemmer():
        return MystemWrapper()

    def __is_equal_results(self, v1, v2):
        print(abs(v1 - v2) < 1e-10)
        self.assertTrue(abs(v1 - v2) < 1e-10)

    def __test_core(self, res_version, synonyms=None,
                    eval_mode=EvaluationModes.Extraction,
                    check_results=True):
        assert(isinstance(res_version, ResultVersions))
        assert(isinstance(synonyms, SynonymsCollection) or synonyms is None)
        assert(isinstance(eval_mode, EvaluationModes))
        assert(isinstance(check_results, bool))

        # Initializing synonyms collection.
        if synonyms is None:
            # This is a default collection which we used
            # to provide the results in `f1_rusentrel_v11_results`.
            stemmer = self.__create_stemmer()
            actual_synonyms = RuSentRelSynonymsCollectionProvider.load_collection(stemmer=stemmer,
                                                                                  version=self.__rusentrel_version)
        else:
            actual_synonyms = synonyms

        # Setup an experiment labels formatter.
        labels_formatter = ExperimentRuSentiFramesLabelsFormatter()

        # Iter cmp opinions.
        cmp_pairs_iter = OpinionCollectionsToCompareUtils.iter_comparable_collections(
            doc_ids=ZippedResultsIOUtils.iter_doc_ids(res_version),
            read_etalon_collection_func=lambda doc_id: OpinionCollection(
                opinions=RuSentRelOpinionCollection.iter_opinions_from_doc(
                    doc_id=doc_id,
                    labels_fmt=labels_formatter),
                synonyms=actual_synonyms,
                error_on_duplicates=False,
                error_on_synonym_end_missed=True),
            read_result_collection_func=lambda doc_id: OpinionCollection(
                opinions=ZippedResultsIOUtils.iter_doc_opinions(
                    doc_id=doc_id,
                    result_version=res_version,
                    labels_formatter=labels_formatter),
                synonyms=actual_synonyms,
                error_on_duplicates=False,
                error_on_synonym_end_missed=False))

        # getting evaluator.
        evaluator = TwoClassEvaluator(eval_mode=eval_mode)

        # evaluate every document.
        logged_cmp_pairs_it = progress_bar_iter(cmp_pairs_iter, desc="Evaluate", unit='pairs')
        result = evaluator.evaluate(cmp_pairs=logged_cmp_pairs_it)
        assert(isinstance(result, TwoClassEvalResult))

        # calculate results.
        result.calculate()

        # logging all the result information.
        for doc_id, doc_info in result.iter_document_results():
            print("{}:\t{}".format(doc_id, doc_info))
        print("------------------------")
        print(str(result.TotalResult))
        print("------------------------")

        # Display cmp tables (optionally).
        if self.__display_cmp_table:
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                for doc_id, df_cmp_table in result.iter_dataframe_cmp_tables():
                    assert(isinstance(df_cmp_table, DocumentCompareTable))
                    print("{}:\t{}\n".format(doc_id, df_cmp_table.DataframeTable))
            print("------------------------")

        if check_results:
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

    def test_rsr_ra_20_large_neut_cnn(self):
        self.__test_core(ResultVersions.CNNRsrRa20LargeNeut)

    def test_classification(self):
        self.__test_core(ResultVersions.SelfTestClassification,
                         eval_mode=EvaluationModes.Classification)


if __name__ == '__main__':
    unittest.main()
