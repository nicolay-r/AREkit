from arekit.common.news_parse_options import NewsParseOptions


class RuAttitudesParseOptions(NewsParseOptions):

    def __init__(self, stemmer, frame_variants_collection):
        super(RuAttitudesParseOptions, self).__init__(parse_entities=True,
                                                      stemmer=stemmer,
                                                      frame_variants_collection=frame_variants_collection)
