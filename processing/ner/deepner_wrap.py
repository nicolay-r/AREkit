from core.processing.ner.base import NamedEntityRecognition
from core.source.entity import Entity
import requests
import json


class DeepNERWrap(NamedEntityRecognition):

    host = "localhost"
    port = 5000
    headers = {'content-type': 'application/json'}

    def __init__(self):
        self.url = "http://{}:{}/ner".format(self.host, self.port)

    def extract(self, text):
        assert(isinstance(text, unicode))

        payload = {"text": text}
        response = requests.post(self.url,
                                 data=json.dumps(payload),
                                 headers=self.headers,
                                 verify=False)
        data = response.json()

        # TODO: Connect with entities.

        return data['tokens'], data['tags']
