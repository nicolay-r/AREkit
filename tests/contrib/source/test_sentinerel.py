import unittest

from arekit.contrib.source.brat.news import BratDocument
from arekit.contrib.source.brat.relation import BratRelation
from arekit.contrib.source.brat.sentence import BratSentence
from arekit.contrib.source.sentinerel.reader import SentiNerelDocReader


class TestRead(unittest.TestCase):

    def test(self):
        news = SentiNerelDocReader.read_document(filename="2070_text", doc_id=0)
        assert(isinstance(news, BratDocument))
        print("Sentences Count:", news.SentencesCount)
        for sentence in news.iter_sentences():
            assert(isinstance(sentence, BratSentence))
            print(sentence.Text.strip())
            for entity, bound in sentence.iter_entity_with_local_bounds():
                print("{}: ['{}',{}, {}]".format(
                    entity.ID, entity.Value, entity.Type,
                    "-".join([str(bound.Position), str(bound.Position+bound.Length)])))

        print()

        for brat_relation in news.Relations:
            assert(isinstance(brat_relation, BratRelation))
            print(brat_relation.SourceID, brat_relation.TargetID, brat_relation.Type)