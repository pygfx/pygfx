class AnimationClip:

    def __init__(self, name="", duration=-1, tracks=None) -> None:
        self.name = name
        self.duration = duration
        self.tracks = tracks or []

        if self.duration < 0:
            self.duration = self.reset_duration()

    def reset_duration(self):
        duration = 0
        for track in self.tracks:
            max_time = track.times[-1]
            duration = max(max_time, duration)
        return duration

    def update(self, time):
        for track in self.tracks:
            track.update(time)
