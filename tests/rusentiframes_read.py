#!/usr/bin/python
from reader.collection import RuSentiFramesCollection
from reader.variants.collection import FrameVariantsCollection
from reader.variants.variant import FrameVariant

frames = RuSentiFramesCollection.from_json("collection.json")

for frame_id in frames.iter_frames_ids():
    # id
    print("Frame: {}".format(frame_id))
    # titles
    print("Titles: {}".format(",".join(frames.get_frame_titles(frame_id))))
    # variants
    print("Variants: {}".format(",".join(frames.get_frame_variants(frame_id))))
    # roles
    for role in frames.get_frame_roles(frame_id):
        print("Role: {}".format(" -- ".join([role.Source, role.Description])))
    # states
    for state in frames.get_frame_states(frame_id):
        print("State: {}".format(",".join([state.Role, state.Label.to_str(), str(state.Prob)])))
    # polarity
    for polarity in frames.get_frame_polarities(frame_id):
        print("Polarity: {}".format(",".join([polarity.Source, polarity.Destination, polarity.Label.to_str()])))

    has_a0_a1_pol = frames.try_get_frame_polarity(frame_id, role_src="a0", role_dest="a1")
    print("Has a0->a1 polarity: {}".format(has_a0_a1_pol is not None))


# frame variants.
frame_variants = FrameVariantsCollection.from_iterable(
    variants_with_id=frames.iter_frame_id_and_variants())
frame_variant = frame_variants.get_variant_by_value("хвалить")
assert(isinstance(frame_variant, FrameVariant))
print("FrameVariantValue: {}".format(frame_variant.get_value()))
print("FrameID: {}".format(frame_variant.FrameID))
