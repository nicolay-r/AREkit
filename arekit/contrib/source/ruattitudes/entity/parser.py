from arekit.contrib.source.brat.entities.parser import BratTextEntitiesParser


class RuAttitudesTextEntitiesParser(BratTextEntitiesParser):

    def __init__(self, **kwargs):
        super(RuAttitudesTextEntitiesParser, self).__init__(partitioning="terms", **kwargs)
