import unittest

from arekit.contrib.networks.types import ModelNamesService


class TestModelNamesService(unittest.TestCase):

    def test(self):
        model_name_type = ModelNamesService.get_type_by_name(u'cnn')

        print model_name_type

        for name in ModelNamesService.iter_supported_names():
            print name,


if __name__ == '__main__':
    unittest.main()
