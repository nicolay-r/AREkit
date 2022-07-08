import unittest

from arekit.common.experiment.data_type import DataType
from arekit.common.folding.fixed import FixedFolding
from arekit.common.folding.types import FoldingType
from arekit.contrib.experiment_rusentrel.factory import create_folding
from arekit.contrib.experiment_rusentrel.types import ExperimentTypes
from arekit.contrib.source.ruattitudes.io_utils import RuAttitudesVersions
from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions


class TestFoldings(unittest.TestCase):

    def test_rusentrel_folding(self):
        folding = create_folding(exp_type=ExperimentTypes.RuAttitudes,
                                 rusentrel_folding_type=FoldingType.CrossValidation,
                                 rusentrel_version=RuSentRelVersions.V11,
                                 ruattitudes_version=RuAttitudesVersions.V20LargeNeut,
                                 ra_doc_id_func=lambda doc_id: doc_id)

        for i, _ in enumerate([0]):
            print("---")
            print("Folding: {}".format(i))
            d = folding.fold_doc_ids_set()
            for k, v in d.items():
                print("{}: {} ({})".format(k, len(v), v[:50]))

    def test_fixed_folding_with_intersection(self):
        fixed_folding = FixedFolding.from_parts(
            {DataType.Train: [1], DataType.Test: [1, 2], DataType.Etalon: [3, 2]})
        d = fixed_folding.fold_doc_ids_set()
        for k, v in d.items():
            print("{}: {} ({})".format(k, len(v), v[:50]))


if __name__ == '__main__':
    unittest.main()
