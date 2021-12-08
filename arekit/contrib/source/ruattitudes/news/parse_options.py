from arekit.common.text.options import NewsParseOptions

# TODO. Why we need to declare this in sources.
# TODO. This information is not related to the particular source.
# TODO. And hence it should be removed from here.
class RuAttitudesParseOptions(NewsParseOptions):

    def __init__(self, stemmer, frame_variants_collection):
        super(RuAttitudesParseOptions, self).__init__(parse_entities=True,
                                                      stemmer=stemmer,
                                                      frame_variants_collection=frame_variants_collection)
