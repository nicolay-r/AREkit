import logging
import pandas as pd
import random

from arekit.common.experiment import const
from arekit.common.experiment.input.providers.label.base import LabelProvider


logger = logging.getLogger(__name__)


class SampleRowBalancerHelper(object):

    # region private methods

    @staticmethod
    def __get_class(df, uint_label):
        return df[df[const.LABEL] == uint_label]

    @staticmethod
    def __get_largest_class_size(df, uint_labels):

        sizes = [len(SampleRowBalancerHelper.__get_class(df=df, uint_label=uint_label))
                 for uint_label in uint_labels]

        return max(sizes)

    @staticmethod
    def __fill_blank(label_df, blank_df, seed=1):
        """
        Composes a DataFrame which has the same amount of examples as one with 'other_label'
        """
        random.seed(seed)
        labels_count = len(label_df)

        if labels_count > 0:
            for row_index in xrange(len(blank_df)):
                keep_row_index = row_index if row_index < labels_count else random.randint(0, labels_count - 1)
                row = label_df.iloc[keep_row_index]
                for column, value in row.iteritems():
                    blank_df.at[row_index, column] = value

        return blank_df

    # endregion

    @staticmethod
    def balance_oversampling(df, create_blank_df, label_provider):
        """
        Balancing related dataframe by amount of examples per class
        create_blank_df: func(size) -> df
        """
        assert(isinstance(df, pd.DataFrame))
        assert(isinstance(label_provider, LabelProvider))
        assert(callable(create_blank_df))

        output_labels = label_provider.OutputLabelsUint

        class_size = SampleRowBalancerHelper.__get_largest_class_size(df=df, uint_labels=output_labels)

        balanced = [SampleRowBalancerHelper.__fill_blank(
                        label_df=SampleRowBalancerHelper.__get_class(df=df, uint_label=label),
                        blank_df=create_blank_df(class_size))
                    for label in output_labels]

        balanced_df = pd.concat(balanced)

        balanced_df.sort_values(by=[const.ID],
                                inplace=True,
                                ascending=True)

        return balanced_df

    # endregion