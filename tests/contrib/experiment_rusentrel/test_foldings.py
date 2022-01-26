import unittest

from arekit.common.folding.types import FoldingType
from arekit.contrib.experiment_rusentrel.factory import create_folding
from arekit.contrib.experiment_rusentrel.types import ExperimentTypes
from arekit.contrib.source.ruattitudes.io_utils import RuAttitudesVersions
from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions


class TestFoldings(unittest.TestCase):

    def test(self):
        folding = create_folding(exp_type=ExperimentTypes.RuSentRelWithRuAttitudes,
                                 rusentrel_folding_type=FoldingType.CrossValidation,
                                 rusentrel_version=RuSentRelVersions.V11,
                                 ruattitudes_version=RuAttitudesVersions.V20LargeNeut)

        print("States: {}".format(folding.StatesCount))
        for i, _ in enumerate(folding.iter_states()):
            print("---")
            print("Folding: {}".format(i))
            d = folding.fold_doc_ids_set()
            for k, v in d.items():
                print("{}: {} ({})".format(k, len(v), v[:50]))


if __name__ == '__main__':
    unittest.main()
