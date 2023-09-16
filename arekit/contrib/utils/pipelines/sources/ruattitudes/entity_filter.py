from arekit.common.entities.types import OpinionEntityType
from arekit.contrib.utils.entities.filter import EntityFilter


class RuAttitudesEntityFilter(EntityFilter):
    """ This is a task-specific filter, which is applicable of entity types proposed
        by the OntoNotesV5 resource: https://catalog.ldc.upenn.edu/LDC2013T19
        We consider only a short list related to the sentiment attitude extraction task.
    """

    supported = ["GPE", "PERSON", "LOCAL", "GEO", "ORG"]

    def is_ignored(self, entity, e_type):

        if e_type == OpinionEntityType.Subject:
            return entity.Type not in RuAttitudesEntityFilter.supported
        if e_type == OpinionEntityType.Object:
            return entity.Type not in RuAttitudesEntityFilter.supported
        else:
            return True
