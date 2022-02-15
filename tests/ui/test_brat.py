import json
import unittest
import pandas as pd
from os.path import join, dirname, realpath

from arekit.common.data import const
from arekit.common.entities.base import Entity
from arekit.common.news.entity import DocumentEntity


class TestBratEmbedding(unittest.TestCase):

    current_dir = dirname(realpath(__file__))
    DATA_DIR = join(current_dir, "data")

    @staticmethod
    def __iter_tsv_gzip(input_file):
        """Reads a tab separated value file."""
        df = pd.read_csv(input_file,
                         compression='gzip',
                         sep='\t',
                         encoding='utf-8')

        for row_index, _ in enumerate(df[const.ID]):
            yield df.iloc[row_index]

    @staticmethod
    def create_relation_types():
        neg = {
            "type": 'NEG',
            "labels": ['NEG'],
            "dashArray": '3,3',
            "color": 'red',
            "args": [
                {"role": 'Subject', "targets": ['Entity']},
                {"role": 'Object', "targets": ['Entity']}
            ]
        }

        pos = {
            "type": 'POS',
            "labels": ['POS'],
            "dashArray": '3,3',
            "color": 'green',
            "args": [
                {"role": 'Subject', "targets": ['Entity']},
                {"role": 'Object', "targets": ['Entity']}
            ]
        }

        return [neg, pos]

    @staticmethod
    def create_entity_types():
        entity_types = []
        person = {
            "type": 'Entity',
            "labels": ['Entity', 'E'],
            "bgColor": '#7fa2ff',
            "borderColor": 'darken'
        }
        entity_types.append(person)
        return entity_types

    @staticmethod
    def handle_document_sentences(sentences, entities):
        assert(isinstance(sentences, dict))
        assert(isinstance(entities, dict))

        e_doc_id = 0
        for s_ind, sentence in sentences.items():

            text_terms = sentence.split(' ')
            for e_ind in entities[s_ind]:
                text_terms[e_ind] = DocumentEntity(value="E", e_type="entity", id_in_doc=e_doc_id, group_index=None)
                e_doc_id += 1

            # Update sentence contents.
            sentences[s_ind] = text_terms

    @staticmethod
    def extract_entities_from_text(text_terms):
        """ ['T1', 'Person', [[0, 11]]]
        """
        assert(isinstance(text_terms, list))

        char_ind = 0
        entities = []
        for term in text_terms:
            if isinstance(term, DocumentEntity):

                t_from = char_ind
                t_to = t_from + len(term.Value)
                entry = ["T{}".format(term.IdInDocument), 'Entity', [[t_from, t_to]]]

                entities.append(entry)

                # update to next
                char_ind = t_to
            else:
                char_ind += len(term)

            # Considering sep
            char_ind += 1

        return entities

    def extract_relations(self, relations, result_data_file):
        """ ['R1', 'N1', [['Subject', 'T2'], ['Object', 'T1']]],
        """
        assert(isinstance(result_data_file, str))

        brat_rels = []
        for r_ind, row in enumerate(self.__iter_tsv_gzip(result_data_file)):

            rel_data = relations[r_ind]

            rel_id = 'R{}'.format(r_ind)
            rel_type = "POS"        # TODO. temporary
            s_ind = 'T{}'.format(rel_data[1])
            t_ind = 'T{}'.format(rel_data[2])

            brat_rels.append([rel_id, rel_type, [['Subject', s_ind], ['Object', t_ind]]])

        return brat_rels

    def create_data(self, input_samples_file):
        assert(isinstance(input_samples_file, str))

        sentences = dict()
        entities = dict()
        relations = []

        for row in self.__iter_tsv_gzip(input_samples_file):

            sent_ind = row['sent_ind']
            relations.append([sent_ind, row['s_ind'], row['t_ind']])

            if sent_ind in sentences:
                continue

            # Provide sentence.
            sentences[sent_ind] = row['text_a']
            entities[sent_ind] = [int(ind) for ind in row['entities'].split(',')]

        # Handle sentences
        self.handle_document_sentences(sentences=sentences, entities=entities)

        # Handle relations
        for r_ind, r_data in enumerate(relations):

            sent_ind = r_data[0]

            e_src = sentences[sent_ind][r_data[1]]
            e_tgt = sentences[sent_ind][r_data[2]]

            assert(isinstance(e_src, DocumentEntity))
            assert(isinstance(e_tgt, DocumentEntity))

            r_data[1] = e_src.IdInDocument
            r_data[2] = e_tgt.IdInDocument

        # Join all the sentences withing a single list of terms.
        doc_terms = []
        for _, s_terms in sentences.items():
            doc_terms.extend(s_terms)

        return doc_terms, relations

    def test(self):

        #################
        # Declaring data.
        result_data = "out.tsv.gz"
        opinions_data = "opinion-test-0.tsv.gz"
        samples_data = "sample-test-0.tsv.gz"
        brat_url = "http://localhost:8001/"

        template_source = join(self.DATA_DIR, "template.html")
        result_data_source = join(self.DATA_DIR, result_data)

        # Loading template file.
        with open(template_source, "r") as templateFile:
            template = templateFile.read()

        # Composing whole output document text.
        samples_data_path = join(self.DATA_DIR, samples_data)
        text_terms, relations = self.create_data(samples_data_path)
        text = " ".join([t.Value if isinstance(t, Entity) else t for t in text_terms])

        # Filling coll data.
        coll_data = dict()
        coll_data['entity_types'] = self.create_entity_types()
        coll_data['relation_types'] = self.create_relation_types()

        # Filling doc data.
        doc_data = dict()
        doc_data['text'] = text
        doc_data['entities'] = self.extract_entities_from_text(text_terms)
        doc_data['relations'] = self.extract_relations(relations=relations, result_data_file=result_data_source)

        print(doc_data)

        # Replace template placeholders.
        template = template.replace("$____COL_DATA_SEM____", json.dumps(coll_data))
        template = template.replace("$____DOC_DATA_SEM____", json.dumps(doc_data))
        template = template.replace("$____TEXT____", text)
        template = template.replace("$____BRAT_URL____", brat_url)

        with open(join(self.DATA_DIR, "output.html"), "w") as output:
            output.write(template)


if __name__ == '__main__':
    unittest.main()
