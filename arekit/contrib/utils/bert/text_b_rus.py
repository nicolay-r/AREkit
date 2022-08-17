from enum import Enum


class BertTextBTemplates(Enum):
    """
    Default, based on COLA, but includes an extra text_b.
        text_b: Pseudo-sentence w/o S.P (S.P -- sentiment polarity)
        text_b: Question w/o S.P (S.P -- sentiment polarity)

    Multilabel variant

    Notation were taken from paper:
    https://www.aclweb.org/anthology/N19-1035.pdf
    """

    NLI = '{subject} к {object} в контексте : << {context} >>'

    QA = 'Что вы думаете по поводу отношения {subject} к {object} в контексте : << {context} >> ?'
