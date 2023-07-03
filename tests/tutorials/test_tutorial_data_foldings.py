import unittest

from arekit.common.experiment.data_type import DataType
from arekit.common.folding.base import BaseDataFolding
from arekit.common.folding.fixed import FixedFolding
from arekit.common.folding.nofold import NoFolding
from arekit.contrib.utils.cv.doc_stat.sentence import SentenceBasedDocumentStatGenerator
from arekit.contrib.utils.cv.splitters.default import SimpleCrossValidationSplitter
from arekit.contrib.utils.cv.splitters.statistical import StatBasedCrossValidationSplitter
from arekit.contrib.utils.cv.two_class import TwoClassCVFolding
from tests.tutorials.test_tutorial_pipeline_text_opinion_annotation import FooDocumentProvider


class DataFolding(unittest.TestCase):

    def show_folding(self, folding, doc_ids):
        assert(isinstance(folding, BaseDataFolding))
        split_dict = folding.fold_doc_ids_set(doc_ids=doc_ids)
        for data_type, doc_ids in split_dict.items():
            print(data_type, doc_ids)

    def test(self):

        parts = {
            DataType.Train: [0, 1, 2, 3],
            DataType.Test: [4, 5, 6, 7]
        }

        fixed_folding = FixedFolding()
        print("Fixed folding:")
        self.show_folding(fixed_folding, doc_ids=parts)

        no_folding = NoFolding(data_type=DataType.Train)
        print("No folding:")
        self.show_folding(no_folding, doc_ids=parts[DataType.Train])

        splitter_simple = SimpleCrossValidationSplitter(shuffle=True, seed=1)

        doc_provider = FooDocumentProvider()
        doc_ids = list(range(2))

        splitter_statistical = StatBasedCrossValidationSplitter(
            docs_stat=SentenceBasedDocumentStatGenerator(doc_reader_func=doc_provider.by_id),
            doc_ids=doc_ids)

        cv_folding = TwoClassCVFolding(
            supported_data_types=[DataType.Train, DataType.Test],
            cv_count=2,
            splitter=splitter_statistical)

        for state_index, _ in enumerate(cv_folding.iter_states()):
            print("State: ", state_index)
            self.show_folding(cv_folding, doc_ids=doc_ids)


if __name__ == '__main__':
    unittest.main()
