import json
from os.path import dirname, realpath, join

from arekit.common.data import const
from arekit.common.data.input.providers.text.single import BaseSingleTextProvider
from arekit.common.data.storages.base import BaseRowsStorage
from arekit.common.entities.base import Entity
from arekit.common.frames.variants.base import FrameVariant
from arekit.common.news.entity import DocumentEntity
from arekit.contrib.networks.core.input.const import Entities, FrameVariantIndices
from arekit.contrib.networks.core.input.rows_parser import ParsedSampleRow


class BratBackend(object):

    current_dir = dirname(realpath(__file__))
    DATA_DIR = join(current_dir, "data")

    @staticmethod
    def __create_relation_types(relation_color_types):
        assert(isinstance(relation_color_types, dict))

        types = []
        for rel_type, rel_color in relation_color_types.items():
            types.append({
                "type": rel_type,
                "labels": [rel_type],
                "dashArray": '3,3',
                "color": rel_color,
                "args": [
                    {"role": 'Subject', "targets": ['Entity']},
                    {"role": 'Object', "targets": ['Entity']}]
            })

        return types

    @staticmethod
    def __create_object_types(entity_color_types):
        assert(isinstance(entity_color_types, dict))

        entity_types = []
        for ent_type, ent_color in entity_color_types.items():
            entity_types.append({
                "type": ent_type,
                "labels": [ent_type],
                "bgColor": ent_color,
                "borderColor": 'darken'
            })

        return entity_types

    @staticmethod
    def __extract_objects(text_terms):
        """ Entities: ['T1', 'Person', [[0, 11]]]
            Triggers: ['T1', 'Frame', [[12, 21]]]
        """
        assert (isinstance(text_terms, list))

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
    def __extract_relations(relations, result_data):
        assert(isinstance(relations, list))
        assert(isinstance(result_data, BaseRowsStorage))

        brat_rels = []
        for r_ind, row in result_data:

            s_ind, t_ind = relations[r_ind]

            neu = int(row['0'])
            pos = int(row['1'])

            if neu > 0:
                continue

            rel_id = 'R{}'.format(r_ind)
            rel_type = "POS" if pos > 0 else "NEG"

            brat_rels.append([rel_id, rel_type, [['Subj', 'T{}'.format(s_ind)],
                                                 ['Obj', 'T{}'.format(t_ind)]]])

        return brat_rels

    @staticmethod
    def __to_terms(sentences_data):
        assert (isinstance(sentences_data, dict))

        sentence_terms = {}

        e_doc_id = 0
        for s_ind, sent_data in sentences_data.items():

            text_terms = sent_data[BaseSingleTextProvider.TEXT_A]
            for i, e_ind in enumerate(sent_data[Entities]):
                sentence_entity_values = sent_data[const.ENTITY_VALUES]
                sentence_entity_types = sent_data[const.ENTITY_TYPES]

                text_terms[e_ind] = DocumentEntity(
                    value=sentence_entity_values[i],
                    e_type=sentence_entity_types[i],
                    id_in_doc=e_doc_id,
                    group_index=None)

                e_doc_id += 1

            for i, f_ind in enumerate(sent_data[FrameVariantIndices]):
                value = text_terms[f_ind]
                text_terms[f_ind] = FrameVariant(text=value, frame_id="0")

            # Update sentence contents.
            sentence_terms[s_ind] = text_terms

        return sentence_terms

    @staticmethod
    def __iter_relations(sentences_data):
        assert(isinstance(sentences_data, dict))
        for s_ind, r_data in sentences_data.items():
            for relation in sentences_data[s_ind]["relations"]:
                terms = sentences_data[s_ind][BaseSingleTextProvider.TEXT_A]
                s_obj = terms[relation[0]]
                t_obj = terms[relation[1]]
                yield [s_obj.IdInDocument, t_obj.IdInDocument]

    def __create_data(self, samples):
        assert(isinstance(samples, BaseRowsStorage))

        sent_data_cols = [BaseSingleTextProvider.TEXT_A,
                          const.ENTITY_VALUES,
                          const.ENTITY_TYPES,
                          Entities,
                          FrameVariantIndices]

        sentences_data = dict()

        for _, row in samples:

            parsed = ParsedSampleRow.parse(row)

            sent_ind = parsed[const.SENT_IND]
            has_row = sent_ind in sentences_data
            s_data = {"relations": []} if not has_row else sentences_data[sent_ind]
            s_data["relations"].append([parsed[const.S_IND], parsed[const.T_IND]])

            if has_row:
                continue

            for col in sent_data_cols:
                s_data[col] = parsed[col]

            sentences_data[sent_ind] = s_data

        # Handle sentences
        sentences_terms = self.__to_terms(sentences_data=sentences_data)
        relations = list(self.__iter_relations(sentences_data=sentences_data))

        # Provide sentence endings.
        for _, terms in sentences_terms.items():
            terms.append('\n')

        # Join all the sentences within a single list of terms.
        doc_terms = []
        for s_terms in sentences_terms.values():
            doc_terms.extend(s_terms)

        return doc_terms, relations

    # TODO. Process text back via pipeline.
    @staticmethod
    def __term_to_text(term):
        if isinstance(term, Entity):
            return term.Value
        if isinstance(term, FrameVariant):
            return term.get_value()
        return term

    def __to_data(self, samples, result, obj_color_types, rel_color_types):
        assert(isinstance(samples, BaseRowsStorage))
        assert(isinstance(result, BaseRowsStorage))

        # Composing whole output document text.
        text_terms, relations = self.__create_data(samples)
        text = " ".join([self.__term_to_text(t) for t in text_terms])

        # Filling coll data.
        coll_data = dict()
        coll_data['entity_types'] = self.__create_object_types(obj_color_types)
        coll_data['relation_types'] = self.__create_relation_types(rel_color_types)

        # Filling doc data.
        doc_data = dict()
        doc_data['text'] = text
        doc_data['entities'] = self.__extract_objects(text_terms)
        doc_data['relations'] = self.__extract_relations(relations=relations,
                                                         result_data=result)

        return text, coll_data, doc_data

    def to_html(self, obj_color_types, rel_color_types,
                samples_data_filepath, result_data_filepath,
                brat_url="http://localhost:8001/"):

        text, coll_data, doc_data = self.__to_data(
            samples=BaseRowsStorage.from_tsv(samples_data_filepath, col_types={'frames': str}),
            result=BaseRowsStorage.from_tsv(result_data_filepath),
            obj_color_types=obj_color_types,
            rel_color_types=rel_color_types,
            )

        # Loading template file.
        template_source = join(self.current_dir, "brat_template.html")
        with open(template_source, "r") as templateFile:
            template = templateFile.read()

        # Replace template placeholders.
        template = template.replace("$____COL_DATA_SEM____", json.dumps(coll_data))
        template = template.replace("$____DOC_DATA_SEM____", json.dumps(doc_data))
        template = template.replace("$____TEXT____", text)
        template = template.replace("$____BRAT_URL____", brat_url)

        return template
