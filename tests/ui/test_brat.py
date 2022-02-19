import json
import unittest
import pandas as pd
from os.path import join, dirname, realpath

from arekit.common.data import const
from arekit.common.data.input.providers.text.single import BaseSingleTextProvider
from arekit.common.entities.base import Entity
from arekit.common.frames.variants.base import FrameVariant
from arekit.common.news.entity import DocumentEntity


class TestBratEmbedding(unittest.TestCase):

    current_dir = dirname(realpath(__file__))
    DATA_DIR = join(current_dir, "data")

    entity_types = ["ORG", "GPE", "PERSON"]

    @staticmethod
    def __iter_tsv_gzip(input_file):
        """Reads a tab separated value file."""
        df = pd.read_csv(input_file,
                         compression='gzip',
                         sep='\t',
                         encoding='utf-8',
                         dtype={'frames': str})

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
    def create_object_types():
        entity_types = []
        org = {"type": 'ORG', "labels": ['ORG'], "bgColor": '#7fa2ff', "borderColor": 'darken'}
        person = {"type": 'PERSON', "labels": ['PERSON', 'PER'], "bgColor": '#7fa200', "borderColor": 'darken'}
        gpe = {"type": 'GPE', "labels": ['GPE'], "bgColor": '#7f00ff', "borderColor": 'darken'}
        frame = {"type": 'Frame', "labels": ['Frame', 'F'], "bgColor": '#00a2ff', "borderColor": 'darken'}
        entity_types.append(org)
        entity_types.append(person)
        entity_types.append(gpe)
        entity_types.append(frame)
        return entity_types

    @staticmethod
    def handle_document_sentences(sentences, entity_inds, entity_values, entity_types, frame_inds):
        assert(isinstance(sentences, dict))
        assert(isinstance(entity_inds, dict))
        assert(isinstance(entity_values, dict))
        assert(isinstance(entity_types, dict))
        assert(isinstance(frame_inds, dict))

        e_doc_id = 0
        for s_ind, sentence in sentences.items():

            text_terms = sentence.split(' ')
            for i, e_ind in enumerate(entity_inds[s_ind]):

                sentence_entity_values = entity_values[s_ind]
                sentence_entity_types = entity_types[s_ind]

                text_terms[e_ind] = DocumentEntity(
                    value=sentence_entity_values[i],
                    e_type=sentence_entity_types[i],
                    id_in_doc=e_doc_id,
                    group_index=None)

                e_doc_id += 1

            if s_ind in frame_inds:
                for i, f_ind in enumerate(frame_inds[s_ind]):
                    value = text_terms[f_ind]
                    text_terms[f_ind] = FrameVariant(text=value, frame_id="0")

            # Update sentence contents.
            sentences[s_ind] = text_terms

    @staticmethod
    def extract_objects(text_terms):
        """ Entities: ['T1', 'Person', [[0, 11]]]
            Triggers: ['T1', 'Frame', [[12, 21]]]
        """
        assert(isinstance(text_terms, list))

        entities_count = 0
        for term in text_terms:
            if isinstance(term, DocumentEntity):
                entities_count = max(entities_count, term.IdInDocument)

        frame_ind = entities_count + 1
        char_ind = 0

        objects = []

        for term in text_terms:
            t_from = char_ind

            if isinstance(term, DocumentEntity):
                t_to = t_from + len(term.Value)
                entity = ["T{}".format(term.IdInDocument), term.Type, [[t_from, t_to]]]
                objects.append(entity)
                # update to next
                char_ind = t_to

            elif isinstance(term, FrameVariant):
                value = term.get_value()
                t_to = t_from + len(value)
                frame = ["T{}".format(frame_ind), "Frame", [[t_from, t_to]]]
                frame_ind += 1
                objects.append(frame)
                char_ind = t_to

            else:
                char_ind += len(term)

            # Considering sep
            char_ind += 1

        return objects

    @staticmethod
    def event_types():
        event_types = []
        frame_based_event = {"type": 'Frame',
                             "labels": ['Frame', 'F'],
                             "bgColor": '#00a2ff',
                             "borderColor": 'darken',
                             "args": [{"type": "Subject", "labels": ["SUBJ"]},
                                      {"type": "Object", "labels": ["OBJ"]}]}
        event_types.append(frame_based_event)
        return event_types

    def extract_relations(self, relations, result_data_file):
        """ ['R1', 'N1', [['Subject', 'T2'], ['Object', 'T1']]],
        """
        assert(isinstance(result_data_file, str))

        brat_rels = []
        for r_ind, row in enumerate(self.__iter_tsv_gzip(result_data_file)):

            rel_data = relations[r_ind]

            neu = int(row['0'])
            pos = int(row['1'])

            if neu > 0:
                continue

            rel_id = 'R{}'.format(r_ind)
            rel_type = "POS" if pos > 0 else "NEG"
            s_ind = 'T{}'.format(rel_data[1])
            t_ind = 'T{}'.format(rel_data[2])

            brat_rels.append([rel_id, rel_type, [['Subject', s_ind], ['Object', t_ind]]])

        return brat_rels

    def create_data(self, input_samples_file):
        assert(isinstance(input_samples_file, str))

        sentences = dict()
        entity_inds = dict()
        entity_values = dict()
        entity_types = dict()
        frame_inds = dict()

        relations = []

        for row in self.__iter_tsv_gzip(input_samples_file):

            sent_ind = row[const.SENT_IND]
            relations.append([sent_ind, row[const.S_IND], row[const.T_IND]])

            if sent_ind in sentences:
                continue

            # Provide sentence.
            sentences[sent_ind] = row[BaseSingleTextProvider.TEXT_A]
            entity_inds[sent_ind] = [int(ind) for ind in row["entities"].split(',')]
            entity_values[sent_ind] = row[const.ENTITY_VALUES].split(',')
            entity_types[sent_ind] = row[const.ENTITY_TYPES].split(',')

            if str(row["frames"]) == 'nan':
                continue

            frame_inds[sent_ind] = [int(ind) for ind in row["frames"].split(',')]

        # Handle sentences
        self.handle_document_sentences(sentences=sentences,
                                       entity_inds=entity_inds,
                                       entity_values=entity_values,
                                       entity_types=entity_types,
                                       frame_inds=frame_inds)

        # Handle relations
        for r_ind, r_data in enumerate(relations):

            sent_ind = r_data[0]

            e_src = sentences[sent_ind][r_data[1]]
            e_tgt = sentences[sent_ind][r_data[2]]

            assert(isinstance(e_src, DocumentEntity))
            assert(isinstance(e_tgt, DocumentEntity))

            r_data[1] = e_src.IdInDocument
            r_data[2] = e_tgt.IdInDocument

        # Provide sentence endings.
        for _, sentence in sentences.items():
            sentence.append('\n')

        # Join all the sentences withing a single list of terms.
        doc_terms = []
        for _, s_terms in sentences.items():
            doc_terms.extend(s_terms)

        return doc_terms, relations

    # TODO. Process text back via pipeline.
    @staticmethod
    def term_to_text(term):
        if isinstance(term, Entity):
            return term.Value
        if isinstance(term, FrameVariant):
            return term.get_value()
        return term

    @staticmethod
    def sentence_to_text(sentence_terms):
        assert(isinstance(sentence_terms, list))
        return " ".join([TestBratEmbedding.term_to_text(t) for t in sentence_terms])

    def test(self):

        #################
        # Declaring data.
        result_data = "out.tsv.gz"
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
        text = " ".join([self.term_to_text(t) for t in text_terms])
        objects = self.extract_objects(text_terms)

        # Filling coll data.
        coll_data = dict()
        coll_data['entity_types'] = self.create_object_types()
        coll_data['relation_types'] = self.create_relation_types()
        coll_data['event_types'] = self.event_types()

        # Filling doc data.
        doc_data = dict()
        doc_data['text'] = text
        doc_data['entities'] = objects
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
