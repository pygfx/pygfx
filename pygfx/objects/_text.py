from ._base import WorldObject


class Text(WorldObject):
    """Render text or a collection of text blocks.

    Can be used to render a piece of multi-paragraph text, or to create a collection of text blocks that
    can be individually positioned.

    See :class:``pygfx.TextGeometry`` and :class:``pygfx.MultiTextGeometry`` for details.
    """

    uniform_type = dict(
        WorldObject.uniform_type,
        rot_scale_transform="2x2xf4",
    )

    def _update_object(self):
        # Is called right before the object is drawn
        super()._update_object()
        self.geometry._on_update_object()

    def _update_world_transform(self):
        # Update when the world transform has changed
        super()._update_world_transform()
        # When rendering in screen space, the world transform is used
        # to establish the point in the scene where the text is placed.
        # The only part of the local transform that is used is the
        # position. Therefore, we also keep a transform containing the
        # local rotation and scale, so that these can be applied to the
        # text in screen coordinates.
        # Note that this applies to the whole text, all text blocks rotate around
        # the text-object origin. To rotate text blocks around their own origin,
        # we should probably implement TextBlock.angle.
        matrix = self.local.matrix[:2, :2]
        self.uniform_buffer.data["rot_scale_transform"] = matrix.T
