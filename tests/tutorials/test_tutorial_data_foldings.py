import unittest
from os.path import dirname, join

from arekit.common.experiment.data_type import DataType
from arekit.common.folding.base import BaseDataFolding
from arekit.common.folding.fixed import FixedFolding
from arekit.common.folding.nofold import NoFolding
from arekit.common.folding.united import UnitedFolding
from arekit.contrib.utils.cv.doc_stat.sentence import SentenceBasedDocumentStatGenerator
from arekit.contrib.utils.cv.splitters.default import SimpleCrossValidationSplitter
from arekit.contrib.utils.cv.splitters.statistical import StatBasedCrossValidationSplitter
from arekit.contrib.utils.cv.two_class import TwoClassCVFolding
from tests.tutorials.test_tutorial_pipeline_text_opinion_annotation import FooDocumentOperations


class DataFolding(unittest.TestCase):

    def show_folding(self, folding):
        assert(isinstance(folding, BaseDataFolding))
        split_dict = folding.fold_doc_ids_set()
        for data_type, doc_ids in split_dict.items():
            print(data_type, doc_ids)

    def test(self):

        parts = {
            DataType.Train: [0, 1, 2, 3],
            DataType.Test: [4, 5, 6, 7]
        }

        fixed_folding = FixedFolding.from_parts(parts)
        print("Fixed folding:")
        self.show_folding(fixed_folding)

        no_folding = NoFolding(doc_ids=[10, 15, 20], supported_data_type=DataType.Dev)
        print("No folding:")
        self.show_folding(no_folding)

        united_folding = UnitedFolding([fixed_folding, no_folding])
        print("United folding:")
        self.show_folding(united_folding)

        splitter_simple = SimpleCrossValidationSplitter(shuffle=True, seed=1)

        doc_ops = FooDocumentOperations()
        splitter_statistical = StatBasedCrossValidationSplitter(
            docs_stat=SentenceBasedDocumentStatGenerator(
                lambda doc_id: doc_ops.get_doc(doc_id)),
            docs_stat_filepath_func=lambda: join(dirname(__file__), "data/stat.txt")
        )

        cv_folding = TwoClassCVFolding(
            supported_data_types=[DataType.Train, DataType.Test],
            doc_ids_to_fold=list(range(10)),
            cv_count=2,
            splitter=splitter_statistical)

        for state_index, _ in enumerate(cv_folding.iter_states()):
            print("State: ", state_index)
            self.show_folding(cv_folding)


if __name__ == '__main__':
    unittest.main()
