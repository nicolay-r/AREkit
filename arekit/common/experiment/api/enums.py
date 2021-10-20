from enum import Enum


class BaseDocumentTag(Enum):

    """ Denotes a document that utilized by annotator algorithm in order to
        provide the related labeling of annotated attitudes in it.
        By default, we consider an empty set, so there is no need to utilize annotator.
    """
    Annotate = 1

    """ Denotes a document that utilized in model evaluation process
    """
    Compare = 2
