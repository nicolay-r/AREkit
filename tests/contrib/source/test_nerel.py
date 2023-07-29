import unittest

from tqdm import tqdm

from arekit.contrib.source.brat.news import BratNews
from arekit.contrib.source.brat.relation import BratRelation
from arekit.contrib.source.brat.sentence import BratSentence
from arekit.contrib.source.nerel.io_utils import NerelIOUtils
from arekit.contrib.source.nerel.reader import NerelDocReader


class TestNerelRead(unittest.TestCase):

    def test(self):
        doc_reader = NerelDocReader()
        news = doc_reader.read_document(filename="109230_text", doc_id=0)
        assert(isinstance(news, BratNews))
        print("Sentences Count:", news.SentencesCount)
        for sentence in news.iter_sentences():
            assert(isinstance(sentence, BratSentence))
            print(sentence.Text.strip())
            for entity, bound in sentence.iter_entity_with_local_bounds():
                print("{}: ['{}',{}, {}]".format(
                    entity.ID, entity.Value, entity.Type,
                    "-".join([str(bound.Position), str(bound.Position+bound.Length)])))

        for brat_relation in news.Relations:
            assert(isinstance(brat_relation, BratRelation))
            print(brat_relation.SourceID, brat_relation.TargetID, brat_relation.Type)

    def test_all_documents(self):
        doc_reader = NerelDocReader()
        filenames_by_ids, folding = NerelIOUtils.read_dataset_split()
        for doc_id in tqdm(folding.iter_doc_ids(), total=len(list(folding.iter_doc_ids()))):
            doc_reader.read_document(filename=filenames_by_ids[doc_id], doc_id=0)
