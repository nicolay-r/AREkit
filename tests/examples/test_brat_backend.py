import unittest
from os.path import join, dirname, realpath

from arekit.contrib.experiment_rusentrel.labels.scalers.three import ThreeLabelScaler
from arekit.contrib.experiment_rusentrel.labels.types import ExperimentNegativeLabel, ExperimentPositiveLabel
from examples.brat_backend import BratBackend


class TestBratEmbedding(unittest.TestCase):

    current_dir = dirname(realpath(__file__))
    DATA_DIR = join(current_dir, "data")

    def test(self):

        brat_be = BratBackend()

        labels_scaler = ThreeLabelScaler()

        template = brat_be.to_html(
            result_data_filepath=join(self.DATA_DIR, "out.tsv.gz"),
            samples_data_filepath=join(self.DATA_DIR, "sample-test-0.tsv.gz"),
            label_to_rel={str(labels_scaler.label_to_uint(ExperimentPositiveLabel())): "POS",
                          str(labels_scaler.label_to_uint(ExperimentNegativeLabel())): "NEG"},
            obj_color_types={"ORG": '#7fa2ff',
                             "GPE": "#7fa200",
                             "PERSON": "#7f00ff",
                             "Frame": "#00a2ff"},
            rel_color_types={"POS": "GREEN",
                             "NEG": "RED"},
            brat_url="http://localhost:8001/")

        with open(join(self.DATA_DIR, "output.html"), "w") as output:
            output.write(template)


if __name__ == '__main__':
    unittest.main()
