from arekit.common.folding.base import BaseDataFolding


class FixedFolding(BaseDataFolding):

    def __init__(self, doc_to_datatypes_func, doc_ids_to_fold, supported_data_types):
        assert(callable(doc_to_datatypes_func))

        super(FixedFolding, self).__init__(doc_ids_to_fold=doc_ids_to_fold,
                                           supported_data_types=supported_data_types)

        self.__doc_to_datatypes_func = doc_to_datatypes_func

    @classmethod
    def from_parts(cls, parts):
        """ parts: dict
                dictionary of {data_type: [doc_ids]}
        """
        assert(isinstance(parts, dict))

        doc_to_datatypes = {}
        for data_type, doc_ids in parts.items():
            for doc_id in doc_ids:
                if doc_id not in doc_to_datatypes:
                    doc_to_datatypes[doc_id] = []
                doc_to_datatypes[doc_id].append(data_type)

        return cls(doc_to_datatypes_func=lambda doc_id: doc_to_datatypes[doc_id],
                   doc_ids_to_fold=doc_to_datatypes.keys(),
                   supported_data_types=list(parts.keys()))

    def fold_doc_ids_set(self):

        folded = {}
        for data_type in self._supported_data_types:
            folded[data_type] = []

        for doc_id in self._doc_ids_to_fold_set:
            for data_type in self.__doc_to_datatypes_func(doc_id):
                folded[data_type].append(doc_id)

        return folded
