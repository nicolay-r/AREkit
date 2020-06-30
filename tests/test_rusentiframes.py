# -*- coding: utf-8 -*-

import logging
import sys
import unittest

sys.path.append('../../')

from arekit.common.frame_variants.collection import FrameVariantsCollection
from arekit.common.frame_variants.base import FrameVariant
from arekit.processing.lemmatization.mystem import MystemWrapper
from arekit.source.rusentiframes.collection import RuSentiFramesCollection


class TestRuSentiFrames(unittest.TestCase):

    def test_reading(self):

        # Initializing logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        logging.basicConfig(level=logging.DEBUG)

        stemmer = MystemWrapper()
        frames = RuSentiFramesCollection.read_collection()

        for frame_id in frames.iter_frames_ids():
            # id
            logger.info("Frame: {}".format(frame_id))
            # titles
            logger.info(u"Titles: {}".format(u",".join(frames.get_frame_titles(frame_id))).encode('utf-8'))
            # variants
            logger.info(u"Variants: {}".format(u",".join(frames.get_frame_variants(frame_id))).encode('utf-8'))
            # roles
            for role in frames.get_frame_roles(frame_id):
                logger.info(u"Role: {}".format(u" -- ".join([role.Source, role.Description])).encode('utf-8'))
            # states
            for state in frames.get_frame_states(frame_id):
                logger.info(u"State: {}".format(u",".join([state.Role, state.Label.to_class_str(), str(state.Prob)])).encode('utf-8'))
            # polarity
            for polarity in frames.get_frame_polarities(frame_id):
                logger.info(u"Polarity: {}".format(u",".join([polarity.Source,
                                                              polarity.Destination,
                                                              polarity.Label.to_class_str()])).encode('utf-8'))

            has_a0_a1_pol = frames.try_get_frame_polarity(frame_id, role_src=u"a0", role_dest=u"a1")
            logger.info(u"Has a0->a1 polarity: {}".format(has_a0_a1_pol is not None).encode('utf-8'))

        # frame variants.
        frame_variants = FrameVariantsCollection.create_unique_variants_from_iterable(
            variants_with_id=frames.iter_frame_id_and_variants(),
            stemmer=stemmer)
        frame_variant = frame_variants.get_variant_by_value(u"хвалить")

        assert(isinstance(frame_variant, FrameVariant))

        logger.info(u"FrameVariantValue: {}".format(frame_variant.get_value()).encode('utf-8'))
        logger.info(u"FrameID: {}".format(frame_variant.FrameID).encode('utf-8'))


if __name__ == '__main__':
    unittest.main()
