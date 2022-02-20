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

    SUBJECT_ROLE = "Subj"
    OBJECT_ROLE = "Obj"

    @staticmethod
    def __create_relation_types(relation_color_types, entity_types):
        assert(isinstance(relation_color_types, dict))
        assert(isinstance(entity_types, list))

        types = []
        for rel_type, rel_color in relation_color_types.items():
            types.append({
                "type": rel_type,
                "labels": [rel_type],
                "dashArray": '3,3',
                "color": rel_color,
                "args": [
                    {"role":  BratBackend.SUBJECT_ROLE, "targets": entity_types},
                    {"role":  BratBackend.OBJECT_ROLE, "targets": entity_types}]
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
    def __extract_relations(relations, result_data, label_to_rel):
        assert(isinstance(result_data, BaseRowsStorage))
        assert(isinstance(label_to_rel, dict))

        relations = sorted(relations, key=lambda item: item[0])

        brat_rels = []
        for res_ind, row in result_data:

            rel_id, s_ind, t_ind = relations[res_ind]

            assert(res_ind == rel_id)

            rel_type = None
            for col_label, rel_name in label_to_rel.items():
                if row[col_label] > 0:
                    rel_type = rel_name
                    break

            # Was not found.
            if rel_type is None:
                continue

            brat_rels.append([rel_id, rel_type, [
                [BratBackend.SUBJECT_ROLE, 'T{}'.format(s_ind)],
                [BratBackend.OBJECT_ROLE, 'T{}'.format(t_ind)]
            ]])

        return brat_rels

    @staticmethod
    def __to_terms(doc_data):
        assert (isinstance(doc_data, dict))

        sentence_terms = []

        e_doc_id = 0
        for s_ind in sorted(doc_data):
            sent_data = doc_data[s_ind]
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
            sentence_terms.append(text_terms)

        return sentence_terms

    @staticmethod
    def __iter_relations(doc_data):
        assert(isinstance(doc_data, dict))
        for s_ind in sorted(doc_data):
            r_data = doc_data[s_ind]
            for relation in r_data["relations"]:
                terms = r_data[BaseSingleTextProvider.TEXT_A]
                r_ind = relation[0]
                s_obj = terms[relation[1]]
                t_obj = terms[relation[2]]
                yield [r_ind, s_obj.IdInDocument, t_obj.IdInDocument]

    @staticmethod
    def __iter_docs_data(samples, sent_data_cols):

        def __create_doc_data():
            return dict()

        doc_data = __create_doc_data()
        curr_doc_id = None

        for row_ind, row in samples:

            parsed = ParsedSampleRow.parse(row)
            doc_id = parsed[const.DOC_ID]

            if curr_doc_id is None:
                curr_doc_id = doc_id
            elif curr_doc_id != doc_id:
                yield curr_doc_id, doc_data
                doc_data = __create_doc_data()

            curr_doc_id = doc_id
            sent_ind = parsed[const.SENT_IND]
            has_row = sent_ind in doc_data
            s_data = {"relations": []} if not has_row else doc_data[sent_ind]
            s_data["relations"].append(
                [row_ind, parsed[const.S_IND], parsed[const.T_IND]]
            )

            if has_row:
                continue

            for col in sent_data_cols:
                s_data[col] = parsed[col]

            doc_data[sent_ind] = s_data

        if len(doc_data) > 0:
            yield curr_doc_id, doc_data

    def __extract_data_from_samples(self, samples):
        assert(isinstance(samples, BaseRowsStorage))

        sent_data_cols = [BaseSingleTextProvider.TEXT_A,
                          const.ENTITY_VALUES,
                          const.ENTITY_TYPES,
                          Entities,
                          FrameVariantIndices]

        # Join all the sentences within a single list of terms.
        text_terms = []
        relations = []

        for doc_id, doc_data in self.__iter_docs_data(samples, sent_data_cols=sent_data_cols):

            # Handle sentences
            sentences_terms = self.__to_terms(doc_data=doc_data)
            relations.extend(self.__iter_relations(doc_data=doc_data))

            # Provide sentence endings.
            for sent_terms in sentences_terms:
                sent_terms.append('\n')

            # Document preamble.
            text_terms.extend(["DOC: {}".format(doc_id), '\n'])

            # Document contents.
            for sent_terms in sentences_terms:
                text_terms.extend(sent_terms)

            # Document appendix.
            text_terms.append('\n')

        return text_terms, relations

    # TODO. Process text back via pipeline.
    @staticmethod
    def __term_to_text(term):
        if isinstance(term, Entity):
            return term.Value
        if isinstance(term, FrameVariant):
            return term.get_value()
        return term

    def __to_data(self, samples, result, obj_color_types, rel_color_types, label_to_rel):
        assert(isinstance(obj_color_types, dict))
        assert(isinstance(rel_color_types, dict))
        assert(isinstance(label_to_rel, dict))
        assert(isinstance(samples, BaseRowsStorage))
        assert(isinstance(result, BaseRowsStorage))

        # Composing whole output document text.
        text_terms, relations = self.__extract_data_from_samples(samples)
        text = " ".join([self.__term_to_text(t) for t in text_terms])

        # Filling coll data.
        coll_data = dict()
        coll_data['entity_types'] = self.__create_object_types(obj_color_types)
        coll_data['relation_types'] = self.__create_relation_types(
            relation_color_types=rel_color_types,
            entity_types=list(obj_color_types.keys()))

        # Filling doc data.
        doc_data = dict()
        doc_data['text'] = text
        doc_data['entities'] = self.__extract_objects(text_terms)
        doc_data['relations'] = self.__extract_relations(relations=relations,
                                                         result_data=result,
                                                         label_to_rel=label_to_rel)

        return text, coll_data, doc_data

    def to_html(self, obj_color_types, rel_color_types,
                samples_data_filepath, result_data_filepath,
                label_to_rel, docs_range=None, brat_url="http://localhost:8001/"):
        assert(isinstance(docs_range, tuple) or docs_range is None)
        assert(isinstance(label_to_rel, dict))

        text, coll_data, doc_data = self.__to_data(
            samples=BaseRowsStorage.from_tsv(samples_data_filepath, col_types={'frames': str}),
            result=BaseRowsStorage.from_tsv(result_data_filepath),
            obj_color_types=obj_color_types,
            rel_color_types=rel_color_types,
            label_to_rel=label_to_rel)

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
