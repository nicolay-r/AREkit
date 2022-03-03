import collections
import json
from os.path import dirname, realpath, join

from tqdm import tqdm

from arekit.common.data import const
from arekit.common.data.input.providers.text.single import BaseSingleTextProvider
from arekit.common.data.storages.base import BaseRowsStorage
from arekit.common.entities.base import Entity
from arekit.common.frames.variants.base import FrameVariant
from arekit.common.news.entity import DocumentEntity
from arekit.contrib.networks.core.input.const import Entities, FrameVariantIndices
from arekit.contrib.networks.core.input.rows_parser import ParsedSampleRow
from arekit.processing.text.token import Token
from arekit.processing.text.tokens import Tokens


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

            text_term = BratBackend.__term_to_text(term)
            t_to = t_from + len(text_term)

            if isinstance(term, DocumentEntity):
                entity = ["T{}".format(term.IdInDocument), term.Type, [[t_from, t_to]]]
                objects.append(entity)

            elif isinstance(term, FrameVariant):
                frame = ["T{}".format(frame_ind), "Frame", [[t_from, t_to]]]
                frame_ind += 1
                objects.append(frame)

            # Considering sep
            char_ind = t_to + 1

        return objects

    @staticmethod
    def __iter_predicted_labels(result_data, label_to_rel):
        assert(isinstance(result_data, BaseRowsStorage))
        assert(isinstance(label_to_rel, dict))

        for res_ind, row in result_data:

            rel_type = None
            for col_label, rel_name in label_to_rel.items():
                if row[col_label] > 0:
                    rel_type = rel_name
                    break

            yield res_ind, rel_type

    @staticmethod
    def __iter_sample_labels(samples, label_to_rel):
        assert(isinstance(samples, BaseRowsStorage))

        for row_ind, row in samples:
            str_label = str(row[const.LABEL]) if const.LABEL in row else None
            label = label_to_rel[str_label] if str_label in label_to_rel else None
            yield row_ind, label

    @staticmethod
    def __extract_relations(relations, labels_iter):
        assert(isinstance(relations, list))
        assert(isinstance(labels_iter, collections.Iterable))

        def __rel_id(r_data):
            return r_data[0]

        relations = sorted(relations, key=lambda item: __rel_id(item))

        # Map relation id towards the related index in list.
        id_to_ind = {}
        for r_ind, r_data in enumerate(relations):
            key = r_data[0]
            id_to_ind[key] = r_ind

        brat_rels = []
        for row_id, rel_type in labels_iter:

            if row_id not in id_to_ind:
                continue

            __id = id_to_ind[row_id]
            rel_id, s_ind, t_ind = relations[__id]
            assert(row_id == rel_id)

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

            for i, term in enumerate(text_terms):
                if not isinstance(term, str):
                    continue
                token = Tokens.try_parse(term)
                if token is None:
                    continue
                text_terms[i] = token

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

    def __extract_data_from_samples(self, samples, docs_range):
        assert(isinstance(samples, BaseRowsStorage))
        assert(isinstance(docs_range, tuple) or docs_range is None)

        sent_data_cols = [BaseSingleTextProvider.TEXT_A,
                          const.ENTITY_VALUES,
                          const.ENTITY_TYPES,
                          Entities,
                          FrameVariantIndices]

        # Join all the sentences within a single list of terms.
        text_terms = []
        relations = []

        for doc_id, doc_data in tqdm(self.__iter_docs_data(samples, sent_data_cols=sent_data_cols)):

            # Check whether document to be saved is actually in range.
            if docs_range is not None and not (doc_id >= docs_range[0] and doc_id <= docs_range[1]):
                continue

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
        if isinstance(term, Token):
            return term.get_meta_value()
        return term

    def __to_data(self, samples, result, obj_color_types, rel_color_types, label_to_rel, docs_range):
        assert(isinstance(obj_color_types, dict))
        assert(isinstance(rel_color_types, dict))
        assert(isinstance(label_to_rel, dict))
        assert(isinstance(samples, BaseRowsStorage))
        assert(isinstance(result, BaseRowsStorage) or result is None)

        # Composing whole output document text.
        text_terms, relations = self.__extract_data_from_samples(
            samples=samples,
            docs_range=docs_range)

        text = " ".join([self.__term_to_text(t) for t in text_terms])

        # Defining the source of labels: from result or predefined.
        labels_src = self.__iter_predicted_labels(result, label_to_rel) \
            if result is not None else self.__iter_sample_labels(samples, label_to_rel)

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
        doc_data['relations'] = self.__extract_relations(relations, labels_src)

        return text, coll_data, doc_data

    def to_html(self, obj_color_types, rel_color_types,
                samples_data_filepath, result_data_filepath,
                label_to_rel, docs_range=None, brat_url="http://localhost:8001/"):
        assert(isinstance(docs_range, tuple) or docs_range is None)
        assert(isinstance(label_to_rel, dict))

        text, coll_data, doc_data = self.__to_data(
            samples=BaseRowsStorage.from_tsv(samples_data_filepath, col_types={'frames': str}),
            result=BaseRowsStorage.from_tsv(result_data_filepath) if result_data_filepath is not None else None,
            obj_color_types=obj_color_types,
            rel_color_types=rel_color_types,
            label_to_rel=label_to_rel,
            docs_range=docs_range)

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
