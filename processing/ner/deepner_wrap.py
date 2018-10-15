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

    def extract(self, text, merge=False):
        """
        tags:
            Provides in a format <part>-<type>, where part could be: B, I, O
            and type: GEO, LOC,
        """
        assert(isinstance(text, unicode))

        payload = {"text": text}
        response = requests.post(self.url,
                                 data=json.dumps(payload),
                                 headers=self.headers,
                                 verify=False)
        data = response.json()

        tokens = data['tokens']
        tags = data['tags']

        if (merge):
            return self._merge(tokens, tags), \
                   [self._tag_type(tag) for tag in tags if self._tag_part(tag) == 'B']

        return tokens, tags

    def _merge(self, tokens, tags):
        merged = []
        for i, tag in enumerate(tags):
            part = self._tag_part()
            if part == 'B':
                merged.append([tokens[i]])
            elif part == 'I':
                merged[len(merged)-1].append(tokens[i])
            self._tag_part(tags)
        return merged


    @staticmethod
    def _tag_part(tag):
        assert(isinstance(tag, unicode))
        return tag[:tag.index('-')]

    @staticmethod
    def _tag_type(tag):
        assert(isinstance(tag, unicode))
        return tag[tag.index('-') + 1:]
