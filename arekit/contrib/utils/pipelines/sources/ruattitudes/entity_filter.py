from arekit.common.entities.types import OpinionEntityType
from arekit.contrib.utils.entities.filter import EntityFilter


class RuAttitudesEntityFilter(EntityFilter):
    """ Among all the entities proposed by the OntonotesV5,
     we consider only a short list related to sentiment attutde extraction task.
    """

    supported = ["GPE", "PERSON", "LOCAL", "GEO", "ORG"]

    def is_ignored(self, entity, e_type):

        if e_type == OpinionEntityType.Subject:
            return entity.Type not in RuAttitudesEntityFilter.supported
        if e_type == OpinionEntityType.Object:
            return entity.Type not in RuAttitudesEntityFilter.supported
        else:
            return True
