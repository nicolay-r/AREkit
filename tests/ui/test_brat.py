import json
import unittest
from os.path import join


class TestBratEmbedding(unittest.TestCase):

    def test(self):

        #################
        # Declaring data.
        data_folder = "data"
        result_data = "out.tsv.gz"
        opinions_data = "opinion-test-0.tsv.gz"
        samples_data = "sample-test-0.tsv.gz"
        text = "Ed O'Kelley was the man who shot the man who shot Jesse James."
        bratUrl = "http://localhost:8001/"

        template = None
        template_source = join(data_folder, "template.html")

        # Loading file.
        with open(template_source, "r") as templateFile:
            template = templateFile.read()

        data = {}

        docData = {
            "text": text + text + text + text + text + text + text + text,
            "entities": [
                ['T1', 'Person', [[0, 11]]],
                ['T2', 'Person', [[20, 23]]],
                ['T3', 'Person', [[37, 40]]],
                ['T4', 'Person', [[50, 61]]],
                ['T5', 'Person', [[280, 291]]],
            ]
        }

        collData = {
            "entity_types": [{
               "type": 'Person',
               "labels": ['Person', 'Per'],
               "bgColor": '#7fa2ff',
               "borderColor": 'darken'
            }]
        }

        collData['relation_types'] = [{
                "type": 'Anaphora',
                "labels": ['Anaphora', 'Ana'],
                "dashArray": '3,3',
                "color": 'purple',
                "args": [
                    {"role": 'Anaphor', "targets": ['Person']},
                    {"role": 'Entity', "targets": ['Person']}
            ]
        }]

        docData['relations'] = [
            ['R1', 'NAME1', [['Anaphor', 'T2'], ['Entity', 'T1']]],
            ['R2', 'NAME2', [['Anaphor', 'T3'], ['Entity', 'T5']]]
        ]

        template = template.replace("$____COL_DATA_SEM____", json.dumps(collData))
        template = template.replace("$____DOC_DATA_SEM____", json.dumps(docData))

        template = template.replace("$____TEXT____", text)
        template = template.replace("$____BRAT_URL____", bratUrl)

        with open(join(data_folder, "output.html"), "w") as output:
            output.write(template)


if __name__ == '__main__':
    unittest.main()
