import unittest
from arekit.contrib.networks.enum_name_types import ModelNamesService


class TestModelNamesService(unittest.TestCase):

    def test(self):
        model_name_type = ModelNamesService.get_type_by_name('cnn')

        print(model_name_type)

        for name in ModelNamesService.iter_supported_names():
            print(name, end=' ')


if __name__ == '__main__':
    unittest.main()
