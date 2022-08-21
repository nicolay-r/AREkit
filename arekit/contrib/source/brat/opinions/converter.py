from arekit.common.labels.str_fmt import StringLabelsFormatter
from arekit.common.text_opinions.base import TextOpinion
from arekit.contrib.source.brat.relation import BratRelation


class BratRelationConverter(object):

    @staticmethod
    def to_text_opinion(brat_relation, doc_id, label_formatter):
        """ Converts opinion into document-level referenced opinion
        """
        assert (isinstance(brat_relation, BratRelation))
        assert(isinstance(label_formatter, StringLabelsFormatter))

        return TextOpinion(doc_id=doc_id,
                           text_opinion_id=int(brat_relation.ID),
                           source_id=brat_relation.SourceID,
                           target_id=brat_relation.TargetID,
                           label=label_formatter.str_to_label(brat_relation.Type))
