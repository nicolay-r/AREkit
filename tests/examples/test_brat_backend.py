import unittest
from os.path import join, dirname, realpath

from arekit.contrib.experiment_rusentrel.labels.scalers.three import ThreeLabelScaler
from arekit.contrib.experiment_rusentrel.labels.types import ExperimentNegativeLabel, ExperimentPositiveLabel
from examples.brat_backend import BratBackend


class TestBratEmbedding(unittest.TestCase):

    current_dir = dirname(realpath(__file__))
    DATA_DIR = join(current_dir, "data")

    def __create_template(self, result_data_filepath, samples_data_filepath, labels_scaler, docs_range=(0, 5)):
        brat_be = BratBackend()
        template = brat_be.to_html(
            result_data_filepath=result_data_filepath,
            samples_data_filepath=samples_data_filepath,
            label_to_rel={str(labels_scaler.label_to_uint(ExperimentPositiveLabel())): "POS",
                          str(labels_scaler.label_to_uint(ExperimentNegativeLabel())): "NEG"},
            obj_color_types={"ORG": '#7fa2ff',
                             "GPE": "#7fa200",
                             "PER": "#7f00ff",
                             "LOC": "#5f00aa",
                             "PERSON": "#7f00ff",
                             "GEOPOLIT": "#7fa200",
                             "Frame": "#00a2ff"},
            rel_color_types={"POS": "GREEN",
                             "NEG": "RED"},
            brat_url="http://localhost:8001/",
            docs_range=docs_range)

        return template

    def test_train(self):
        template = self.__create_template(samples_data_filepath=join(self.DATA_DIR, "sample-train-0.tsv.gz"),
                                          result_data_filepath=None,
                                          labels_scaler=ThreeLabelScaler(),
                                          docs_range=(0, 2))

        with open(join(self.DATA_DIR, "output_train.html"), "w") as output:
            output.write(template)

    def test(self):
        template = self.__create_template(samples_data_filepath=join(self.DATA_DIR, "sample-test-0.tsv.gz"),
                                          result_data_filepath=join(self.DATA_DIR, "out.tsv.gz"),
                                          labels_scaler=ThreeLabelScaler())

        with open(join(self.DATA_DIR, "output.html"), "w") as output:
            output.write(template)


if __name__ == '__main__':
    unittest.main()
