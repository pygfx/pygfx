import math
from .interpolant import LinearInterpolant
from ..objects import Event


class AnimationAction:
    """An AnimationAction represents a specific animation that can be played within the AnimationMixer.

    It is a single animation clip that can be played, paused, stopped, faded in/out, and cross-faded with other actions.
    """

    def __init__(self, mixer, clip):
        self._mixer = mixer
        self._clip = clip

        self.weight = 1.0
        self._effective_weight = 1.0

        self.time = 0
        self.time_scale = 1.0
        self._effective_time_scale = 1.0

        self.repetitions = math.inf

        self.paused = False
        self.enabled = True

        self._weight_interpolant = None
        self._time_scale_interpolant = None

    def play(self):
        """Play the action."""
        self._mixer._activate_action(self)
        return self

    def stop(self):
        """Stop the action."""
        self._mixer._deactivate_action(self)
        return self.reset()

    def reset(self):
        """Reset the action."""
        self.paused = False
        self.enabled = True

        self.time = 0

        return self.stop_fading()

    def is_running(self):
        return (
            self.enabled
            and not self.paused
            and self.time_scale != 0
            and self._mixer._is_active_action(self)
        )

    def is_schedule(self):
        # return True when play is called
        return self._mixer._is_active_action(self)

    def set_loop(self, repetitions=math.inf):
        """Set the number of repetitions for the action."""
        self.repetitions = repetitions
        return self

    @property
    def effective_weight(self):
        return self._effective_weight

    def set_effective_weight(self, weight):
        """Set the effective weight for the action."""
        self.weight = weight
        self._effective_weight = weight if self.enabled else 0

        return self.stop_fading()

    def fade_in(self, duration):
        """Fade in the action."""
        return self._schedule_fading(duration, 0, 1)

    def fade_out(self, duration):
        """Fade out the action."""
        return self._schedule_fading(duration, 1, 0)

    def cross_fade_from(self, fade_out_action: "AnimationAction", duration, warp):
        """Cross fade from another action."""
        fade_out_action.fade_out(duration)
        self.fade_in(duration)

        if warp:
            fade_in_duration = self._clip.duration
            fade_out_duration = fade_out_action._clip.duration

            start_end_ratio = fade_out_duration / fade_in_duration
            end_start_ratio = fade_in_duration / fade_out_duration

            fade_out_action.warp(1.0, start_end_ratio, duration)
            self.warp(end_start_ratio, 1.0, duration)
        return self

    def cross_fade_to(self, fade_in_action: "AnimationAction", duration, warp):
        """Cross fade to another action."""
        return fade_in_action.cross_fade_from(self, duration, warp)

    def stop_fading(self):
        """Stop the fading of the action."""
        self._weight_interpolant = None
        return self

    @property
    def effective_time_scale(self):
        return self._effective_time_scale

    def set_effective_time_scale(self, time_scale):
        """Set the effective time scale for the action."""
        self.time_scale = time_scale
        self._effective_time_scale = 0 if self.paused else time_scale
        return self.stop_warping()

    def set_duration(self, duration):
        """Set the duration of the action by adjusting the time scale."""
        self.time_scale = self._clip.duration / duration
        return self.stop_warping()

    def halt(self, duration):
        """Halt the action."""
        return self.warp(self._effective_time_scale, 0, duration)

    def warp(self, start_time_scale, end_time_scale, duration):
        """Warp the action."""
        now = self._mixer._time
        interpolant = self._time_scale_interpolant
        time_scale = self.time_scale

        if interpolant is None:
            times = [now, now + duration]
            values = [start_time_scale, end_time_scale]
            interpolant = LinearInterpolant(times, values)
            self._time_scale_interpolant = interpolant
        else:
            interpolant.parameter_positions[0] = now
            interpolant.parameter_positions[1] = now + duration
            interpolant.sample_values[0] = start_time_scale / time_scale
            interpolant.sample_values[1] = end_time_scale / time_scale

        return self

    def stop_warping(self):
        """Stop the warping of the action."""
        self._time_scale_interpolant = None
        return self

    def _update_weight(self, time):
        weight = 0
        if self.enabled:
            weight = self.weight

            weight_interpolant = self._weight_interpolant
            if weight_interpolant is not None:
                interpolant_value = weight_interpolant(time)
                weight *= interpolant_value

                if time > weight_interpolant.parameter_positions[1]:
                    # self._weight_interpolant = None # clear the interpolant
                    self.stop_fading()

                    if interpolant_value == 0:
                        # faded out, disable
                        self.enabled = False

        self._effective_weight = weight
        return weight

    def _update_time_scale(self, time):
        time_scale = 0
        if not self.paused:
            time_scale = self.time_scale

            time_scale_interpolant = self._time_scale_interpolant
            if time_scale_interpolant is not None:
                interpolant_value = time_scale_interpolant(time)
                time_scale *= interpolant_value
                if time > time_scale_interpolant.parameter_positions[1]:
                    # self._time_scale_interpolant = None # clear the interpolant
                    self.stop_warping()

                    if time_scale == 0:
                        # has halted, pause
                        self.paused = True
                    else:
                        # warp done, apply final time scale
                        self.time_scale = time_scale

        self._effective_time_scale = time_scale
        return time_scale

    def _update_time(self, dt):
        time = self.time + dt
        duration = self._clip.duration
        if duration == 0:
            self.enabled = False
            return time

        if time >= duration:
            self.repetitions -= 1
            if self.repetitions <= 0:
                self.enabled = False
                time = duration
            else:
                time = time % duration
                event = Event(type="loop")
                event.action = self
                self._mixer.dispatch_event(event)

        self.time = time
        return time

    def _schedule_fading(self, duration, weight_now, weight_then):
        now = self._mixer._time
        interpolant = self._weight_interpolant
        if interpolant is None:
            times = [now, now + duration]
            values = [weight_now, weight_then]
            interpolant = LinearInterpolant(times, values)
            self._weight_interpolant = interpolant
        else:
            interpolant.parameter_positions[0] = now
            interpolant.parameter_positions[1] = now + duration
            interpolant.sample_values[0] = weight_now
            interpolant.sample_values[1] = weight_then

        return self

    def _update(self, dt):
        if not self.enabled:
            # update ._effective_weight
            self._update_weight(self._mixer._time)
            return

        # apply time scale
        dt = dt * self._update_time_scale(self._mixer._time)
        clip_time = self._update_time(dt)

        weight = self._update_weight(self._mixer._time)

        if weight > 0:
            for track in self._clip.tracks:
                value = track.interpolation(clip_time)
                self._mixer._accumulate(track.target, track.path, value, weight)
