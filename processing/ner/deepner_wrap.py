from core.processing.ner.base import NamedEntityRecognition
import requests
import json


class DeepNERWrap(NamedEntityRecognition):

    host = "localhost"
    port = 5000
    headers = {'content-type': 'application/json'}
    separator = '-'

    def __init__(self):
        self.url = "http://{}:{}/ner".format(self.host, self.port)

    def _extract_tags(self, terms):
        """
        terms: list
        tags:
            Provides in a format <part>-<type>, where part could be: B, I, O
            and type: GEO, LOC,
        """
        assert(isinstance(terms, list))

        payload = {"terms": terms}
        response = requests.post(self.url,
                                 data=json.dumps(payload),
                                 headers=self.headers,
                                 verify=False)
        data = response.json()

        return data['tags']
