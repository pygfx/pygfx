class AnimationClip:
    """AnimationClip is a collection of tracks that are played together."""

    def __init__(self, name="", duration=-1, tracks=None) -> None:
        self.name = name
        self.duration = duration
        self.tracks = tracks or []

        if self.duration < 0:
            self.duration = self.reset_duration()

    def reset_duration(self):
        """Reset the duration of the clip to the maximum duration of all tracks."""
        duration = 0
        for track in self.tracks:
            max_time = track.times[-1]
            duration = max(max_time, duration)
        return duration
