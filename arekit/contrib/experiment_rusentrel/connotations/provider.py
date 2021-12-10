from arekit.common.frames.connotations.provider import FrameConnotationProvider
from arekit.contrib.source.rusentiframes.collection import RuSentiFramesCollection


class RuSentiFramesConnotationProvider(FrameConnotationProvider):

    def __init__(self, collection):
        assert(isinstance(collection, RuSentiFramesCollection))
        self.__collection = collection

    def try_get_frame_sentiment_polarity(self, frame_id):
        return self.__collection.try_get_frame_polarity(frame_id=frame_id,
                                                        role_src='a0',
                                                        role_dest='a1')