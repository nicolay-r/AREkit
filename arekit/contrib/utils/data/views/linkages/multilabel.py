import numpy as np
import pandas as pd

from arekit.common.data import const
from arekit.common.data.row_ids.multiple import MultipleIDProvider
from arekit.common.labels.scaler.base import BaseLabelScaler
from arekit.contrib.utils.data.views.linkages import utils
from arekit.contrib.utils.data.views.linkages.base import BaseOpinionLinkagesView
from arekit.contrib.utils.data.views.opinions import BaseOpinionStorageView


class MultilableOpinionLinkagesView(BaseOpinionLinkagesView):
    """ View onto source, where each row, related to opinion, has multiple labels.
    """

    def __init__(self, labels_scaler, storage):
        assert(isinstance(labels_scaler, BaseLabelScaler))
        super(MultilableOpinionLinkagesView, self).__init__(ids_provider=MultipleIDProvider(),
                                                            storage=storage)
        self.__labels_scaler = labels_scaler

    # region private methods

    def __get_column_header(self):
        return [str(self.__labels_scaler.label_to_uint(label))
                for label in self.__labels_scaler.ordered_suppoted_labels()]

    def __calculate_label(self, row):
        """
        Using a single row (probabilities by each class)
        """
        labels_prob = [row[label] for label in self.__get_column_header()]
        return self.__labels_scaler.uint_to_label(value=int(np.argmax(labels_prob)))

    # endregion

    # region protected methods

    def _iter_by_opinions(self, linked_df, opinions_view):
        assert(isinstance(linked_df, pd.DataFrame))
        assert(isinstance(opinions_view, BaseOpinionStorageView))

        for _, series in linked_df.iterrows():
            yield utils.compose_opinion_by_opinion_id(
                ids_provider=self._ids_provider,
                sample_id=series[const.ID],
                opinions_view=opinions_view,
                calc_label_func=lambda: self.__calculate_label(series))

    # endregion
