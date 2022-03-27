import unittest

from arekit.contrib.bert.samplers.types import SampleFormattersService, BertSampleProviderTypes
from arekit.contrib.experiment_rusentrel.entities.types import EntityFormattersService, EntityFormatterTypes
from arekit.contrib.experiment_rusentrel.types import ExperimentTypesService, ExperimentTypes


class EnumReadingTest(unittest.TestCase):

    def test(self):

        # 1
        n_exp = 'rsr'
        t_exp = ExperimentTypes.RuSentRel
        t_act = ExperimentTypesService.name_to_type(n_exp)
        n_act = ExperimentTypesService.type_to_name(t_exp)
        assert(t_exp == t_act)
        assert(n_exp == n_act)

        # 2
        n_exp = 'hidden-simple-rus'
        t_exp = EntityFormatterTypes.HiddenSimpleRus
        t_act = EntityFormattersService.name_to_type(n_exp)
        n_act = EntityFormattersService.type_to_name(t_exp)
        assert(t_exp == t_act)
        assert(n_exp == n_act)

        # 3
        n_exp = 'c_m'
        t_exp = BertSampleProviderTypes.CLASSIF_M
        t_act = SampleFormattersService.name_to_type(n_exp)
        n_act = SampleFormattersService.type_to_name(t_exp)
        assert(t_exp == t_act)
        assert(n_exp == n_act)

        supported = list(ExperimentTypesService.iter_names())
        print(supported)


if __name__ == '__main__':
    unittest.main()
