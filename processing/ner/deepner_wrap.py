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

    def extract(self, terms, merge=False):
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

        result_terms = data['tokens']
        tags = data['tags']

        if not merge:
            return result_terms, tags

        merged_terms = self.__merge(result_terms, tags)
        types = [self.__tag_type(tag) for tag in tags if self.__tag_part(tag) == 'B']
        positions = [i for i, tag in enumerate(tags) if self.__tag_part(tag) == 'B']
        return merged_terms, types, positions

    def __merge(self, terms, tags):
        merged = []
        for i, tag in enumerate(tags):
            part = self.__tag_part(tag)
            if part == 'B':
                merged.append([terms[i]])
            elif part == 'I' and len(merged) > 0:
                merged[len(merged)-1].append(terms[i])
        return merged

    @staticmethod
    def __tag_part(tag):
        assert(isinstance(tag, unicode))
        return tag if DeepNERWrap.separator not in tag \
            else tag[:tag.index(DeepNERWrap.separator)]

    @staticmethod
    def __tag_type(tag):
        assert(isinstance(tag, unicode))
        return "" if DeepNERWrap.separator not in tag \
            else tag[tag.index(DeepNERWrap.separator) + 1:]
