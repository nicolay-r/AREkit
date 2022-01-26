from arekit.common.folding.base import BaseDataFolding


class UnitedFolding(BaseDataFolding):

    def __init__(self, foldings):
        assert(isinstance(foldings, list))

        states_count = None

        # Guarantee exact the same states count as passed parameter.
        for folding in foldings:
            assert(isinstance(folding, BaseDataFolding))
            if states_count is None:
                states_count = folding._states
            assert(folding._states == states_count)

        self.__foldings = foldings

        super(UnitedFolding, self).__init__(
            doc_ids_to_fold=UnitedFolding.__iter_all_doc_ids(foldings),
            supported_data_types=list(set(UnitedFolding.__iter_all_data_types(foldings))),
            states_count=states_count)

    @property
    def Name(self):
        return "united"

    @staticmethod
    def __iter_all_doc_ids(foldings):
        for folding in foldings:
            assert(isinstance(folding, BaseDataFolding))
            for doc_id in folding.iter_doc_ids():
                yield doc_id

    @staticmethod
    def __iter_all_data_types(foldings):
        for folding in foldings:
            assert(isinstance(folding, BaseDataFolding))
            for d_type in folding.iter_supported_data_types():
                yield d_type

    @staticmethod
    def __merge(origin, new_data):
        assert(isinstance(origin, dict))
        assert(isinstance(new_data, dict))
        for key, value in new_data.items():
            if key not in origin:
                # Assign list
                origin[key] = value
            else:
                # Combine lists
                origin[key] += value

    def fold_doc_ids_set(self):
        origin = {}
        for folding in self.__foldings:
            assert(isinstance(folding, BaseDataFolding))
            new_data = folding.fold_doc_ids_set()
            self.__merge(origin=origin, new_data=new_data)

        return origin

    def _assign_index(self, i):
        """ Update current index for every folding.
        """
        for folding in self.__foldings:
            assert(isinstance(folding, BaseDataFolding))
            folding._assign_index(i)
