# -*- coding: utf-8 -*-
#!/usr/bin/python
from core.common.frame_variants.collection import FrameVariantsCollection
from core.common.frame_variants.base import FrameVariant
from core.processing.lemmatization.mystem import MystemWrapper
from core.source.rusentiframes.collection import RuSentiFramesCollection


stemmer = MystemWrapper()
frames = RuSentiFramesCollection.read_collection()

for frame_id in frames.iter_frames_ids():
    # id
    print "Frame: {}".format(frame_id)
    # titles
    print u"Titles: {}".format(u",".join(frames.get_frame_titles(frame_id))).encode('utf-8')
    # variants
    print u"Variants: {}".format(u",".join(frames.get_frame_variants(frame_id))).encode('utf-8')
    # roles
    for role in frames.get_frame_roles(frame_id):
        print u"Role: {}".format(u" -- ".join([role.Source, role.Description])).encode('utf-8')
    # states
    for state in frames.get_frame_states(frame_id):
        print u"State: {}".format(u",".join([state.Role, state.Label.to_str(), str(state.Prob)])).encode('utf-8')
    # polarity
    for polarity in frames.get_frame_polarities(frame_id):
        print u"Polarity: {}".format(u",".join([polarity.Source, polarity.Destination, polarity.Label.to_str()])).encode('utf-8')

    has_a0_a1_pol = frames.try_get_frame_polarity(frame_id, role_src=u"a0", role_dest=u"a1")
    print u"Has a0->a1 polarity: {}".format(has_a0_a1_pol is not None).encode('utf-8')

# frame variants.
frame_variants = FrameVariantsCollection.from_iterable(
    variants_with_id=frames.iter_frame_id_and_variants(),
    stemmer=stemmer)
frame_variant = frame_variants.get_variant_by_value(u"хвалить")

assert(isinstance(frame_variant, FrameVariant))

print u"FrameVariantValue: {}".format(frame_variant.get_value()).encode('utf-8')
print u"FrameID: {}".format(frame_variant.FrameID).encode('utf-8')
