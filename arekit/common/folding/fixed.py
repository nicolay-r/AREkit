from arekit.common.folding.base import BaseDataFolding


class FixedFolding(BaseDataFolding):

    @staticmethod
    def _doc_to_datatypes(datatype_to_docs):
        assert(isinstance(datatype_to_docs, dict))

        doc_to_datatypes = {}
        for data_type, doc_ids in datatype_to_docs.items():
            for doc_id in doc_ids:
                if doc_id not in doc_to_datatypes:
                    doc_to_datatypes[doc_id] = []
                doc_to_datatypes[doc_id].append(data_type)

        return doc_to_datatypes

    def fold_doc_ids_set(self, doc_ids):
        assert(isinstance(doc_ids, dict))

        doc_to_datatype = self._doc_to_datatypes(doc_ids)

        folded = {}
        for data_type in doc_ids.keys():
            folded[data_type] = []

        for doc_id in doc_to_datatype.keys():
            for data_type in doc_to_datatype[doc_id]:
                folded[data_type].append(doc_id)

        return folded
