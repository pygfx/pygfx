"""
Audio Visualizer
================

This example demonstrates how to create an audio visualizer.
"""

# sphinx_gallery_pygfx_docs = 'code'
# sphinx_gallery_pygfx_test = 'off'

import numpy as np
import wgpu
import pygfx as gfx
import io
import os
import threading
import sounddevice as sd
import soundfile as sf
import requests
from tqdm import tqdm
from wgpu.gui.auto import WgpuCanvas, run
from pygfx.renderers.wgpu import (
    Binding,
    RenderMask,
    GfxSampler,
    GfxTextureView,
    register_wgpu_render_function,
)
from pygfx.renderers.wgpu.shaders.meshshader import BaseShader


class NumpyCircularBuffer:
    """
    Circular buffer implemented using numpy arrays.
    This is used to store the last N samples of audio data.
    """

    def __init__(self, max_size, data_shape, dtype=np.float32):
        self.max_size = max_size
        self.data_shape = data_shape
        self.buffer = np.zeros((max_size, *data_shape), dtype=dtype)
        self.index = 0

    def append(self, data):
        num_items = data.shape[0]
        if num_items > self.max_size:
            raise ValueError("The input data is larger than the buffer size.")

        # Calculate the insertion index range
        end_index = (self.index + num_items) % self.max_size

        if end_index > self.index:
            self.buffer[self.index : end_index] = data
        else:
            # Wrap-around case
            part1_size = self.max_size - self.index
            self.buffer[self.index :] = data[:part1_size]
            self.buffer[:end_index] = data[part1_size:]

        self.index = end_index

    def get_last_n(self, n):
        start_index = (self.index - n) % self.max_size
        if start_index < 0:
            start_index += self.max_size

        if start_index < self.index:
            return self.buffer[start_index : self.index]
        else:
            return np.concatenate(
                (self.buffer[start_index:], self.buffer[: self.index]), axis=0
            )


class AudioAnalyzer:
    """
    Simple implementation of Audio AnalyserNode in W3C Web Audio API.
    See: https://www.w3.org/TR/webaudio/#AnalyserNode
    """

    def __init__(
        self, fft_size=1024, min_decibels=-100, max_decibels=-30, smoothing_factor=0.8
    ):
        self.fft_size = fft_size
        self.min_decibels = min_decibels
        self.max_decibels = max_decibels
        self.smoothing_factor = smoothing_factor

        # last 32768 samples, 2 channels
        # In order to allow for an increase in fftsize, we should effectively keep around the last 32768 samples
        self._buffer = NumpyCircularBuffer(32768, (2,))

    @property
    def frequency_bin_count(self):
        return self.fft_size // 2

    @property
    def fft_size(self):
        return self._fft_size

    @fft_size.setter
    def fft_size(self, value):
        assert value <= 32768 and value >= 32, "fft_size must be between 32 and 32768"
        assert value & (value - 1) == 0, "fft_size must be a power of 2"
        self._fft_size = value
        self._frequency_data = np.zeros(self.fft_size // 2 + 1, dtype=np.float32)
        self._byte_frequency_data = np.zeros(self.fft_size // 2 + 1, dtype=np.uint8)
        self.__blackman_window = self._get_blackman_window()
        self.__previous_smoothed_data = np.zeros(value // 2 + 1, dtype=np.float32)

    def _get_blackman_window(self):
        a0 = 0.42
        a1 = 0.5
        a2 = 0.08
        n = np.arange(self.fft_size, dtype=np.float32)
        w = (
            a0
            - a1 * np.cos(2 * np.pi * n / self.fft_size)
            + a2 * np.cos(4 * np.pi * n / self.fft_size)
        )  # W3C spec use N not N-1, so we use fft_size not fft_size-1 here, todo: confirm is it correct?
        return w

    def receive_data(self, data):
        self._buffer.append(data)

        # A block of 128 samples-frames is called a render quantum
        # within the same render quantum as a previous call, the current frequency data is not updated with the same data.
        # Instead, the previously computed data is returned.
        # we assume that len(data) always >= 128
        self._byte_frequency_data = None
        self._frequency_data = None

    def get_time_domain_data(self):
        time_domain_data = self._buffer.get_last_n(self.fft_size)
        time_domain_data = np.mean(time_domain_data, axis=1, dtype=np.float32)
        # the data should be already in range -1 to 1, but we clip it just in case of any overflow
        time_domain_data = np.clip(time_domain_data, -1, 1)
        return time_domain_data

    def get_byte_time_domain_data(self):
        time_domain_data = self.get_time_domain_data()
        return np.floor((time_domain_data + 1) * 128).astype(np.uint8)

    def get_frequency_data(self):
        if self._frequency_data is None:
            time_domain_data = self.get_time_domain_data()

            frames_windowed = time_domain_data * self.__blackman_window
            # Perform FFT
            # def _fft(data):
            #     N = len(data)
            #     W = np.exp(-2j * np.pi / N)
            #     X = np.zeros(N // 2 + 1, dtype=complex)
            #     for k in range(N // 2 + 1):
            #         for n in range(N):
            #             X[k] += data[n] * W**(k * n)
            #         X[k] /= N
            #     return X
            fft_result = np.fft.rfft(frames_windowed, n=self.fft_size) / self.fft_size

            # Smooth over time
            smoothing_factor = self.smoothing_factor
            smoothed_data = smoothing_factor * self.__previous_smoothed_data + (
                1 - smoothing_factor
            ) * np.abs(fft_result)

            # Handle non-finite values
            smoothed_data = np.nan_to_num(
                smoothed_data, nan=0.0, posinf=0.0, neginf=0.0
            )

            # Update previous smoothed data
            self.__previous_smoothed_data = smoothed_data

            # Convert to dB
            self._frequency_data = 20 * np.log10(smoothed_data)

        return self._frequency_data

    def get_byte_frequency_data(self):
        if self._byte_frequency_data is None:
            frequency_data = self.get_frequency_data()

            clipped_data = np.clip(frequency_data, self.min_decibels, self.max_decibels)
            scale = 255 / (self.max_decibels - self.min_decibels)
            self._byte_frequency_data = np.floor(
                (clipped_data - self.min_decibels) * scale
            ).astype(np.uint8)

        return self._byte_frequency_data


class AudioPlayer:
    def __init__(self) -> None:
        self._analyzer = None
        # we use 128 samples as a block size as default, it's called a render quantum in W3C spec
        self._block_size = 128
        self._cache_block_size = 50 * 1024
        self._mini_playable_size = 10 * 1024

    @property
    def block_size(self):
        return self._block_size

    @property
    def analyzer(self):
        return self._analyzer

    @analyzer.setter
    def analyzer(self, analyzer):
        self._analyzer = analyzer

    def play(self, path, stream=True):
        if "https://" in str(path) or "http://" in str(path):
            if stream:
                play_func = self._play_stream
            else:
                r = requests.get(path)
                r.raise_for_status()
                path = io.BytesIO(r.content)
                play_func = self._play_local
        else:
            play_func = self._play_local

        play_t = threading.Thread(target=play_func, args=(path,), daemon=True)
        play_t.start()

    def _play_local(self, local_file):
        data, samplerate = sf.read(local_file, dtype=np.float32, always_2d=True)
        block_size = self.block_size

        stream = sd.OutputStream(
            samplerate=samplerate,
            channels=data.shape[1],
            dtype=np.float32,
            blocksize=block_size,
        )

        with stream:
            length = len(data)
            with tqdm(
                total=length / samplerate + 0.001,
                unit="s",
                unit_scale=True,
                desc="Playing",
            ) as pbar:
                for i in range(0, length, block_size):
                    frames_data = data[i : min(i + block_size, length)]
                    stream.write(frames_data)
                    if self.analyzer:
                        self.analyzer.receive_data(frames_data)
                    pbar.update(block_size / samplerate)

    def _play_stream(self, path):
        response = requests.get(path, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))

        audio_data = io.BytesIO()
        bytes_lock = threading.Lock()
        block_data_available = threading.Event()

        def _download_data():
            chunk_size = 1024
            playback_block_size = self._cache_block_size
            mini_playable_size = self._mini_playable_size
            with tqdm(
                total=total_size, unit="B", unit_scale=True, desc="Downloading"
            ) as dbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    # time.sleep(0.05) # simulate slow download
                    with bytes_lock:
                        last_read_pos = audio_data.tell()
                        audio_data.seek(0, os.SEEK_END)
                        audio_data.write(chunk)
                        end_pos = audio_data.tell()
                        audio_data.seek(0)
                        audio_data.seek(last_read_pos)
                    if end_pos - last_read_pos > playback_block_size:
                        block_data_available.set()  # resume playback if buffer is enough
                    elif end_pos - last_read_pos <= mini_playable_size:
                        block_data_available.clear()  # pause playback if not enough data
                    dbar.update(len(chunk))
                block_data_available.set()

        download_data_t = threading.Thread(target=_download_data, daemon=True)
        download_data_t.start()

        # wait for the first block of data to be available, then create the soundFile
        while True:
            try:
                block_data_available.wait()
                with bytes_lock:
                    audio_data.seek(0)
                    audio_file = sf.SoundFile(audio_data, mode="r")
                    break
            except Exception:
                block_data_available.clear()

        block_size = self.block_size

        stream = sd.OutputStream(
            samplerate=audio_file.samplerate,
            channels=audio_file.channels,
            dtype=np.float32,
            blocksize=block_size,
        )

        total_time = audio_file.frames / audio_file.samplerate + 0.001
        with stream:
            with tqdm(
                total=total_time, unit="s", unit_scale=True, desc="Playing"
            ) as pbar:
                while True:
                    block_data_available.wait()
                    with bytes_lock:
                        data = audio_file.read(
                            block_size, dtype=np.float32, always_2d=True
                        )
                    if len(data) > 0:
                        stream.write(data)
                        if self.analyzer:
                            self.analyzer.receive_data(data)
                        pbar.update(len(data) / audio_file.samplerate)
                    else:
                        break


class AudioMaterial(gfx.Material):
    def __init__(
        self, audio_data, fragment_shader_code=None, interpolation="nearest", **kwargs
    ):
        super().__init__(**kwargs)
        self._audio_data = audio_data

        self._interpolation = interpolation
        self._fragment_shader_code = fragment_shader_code

    @property
    def audio_data(self):
        return self._audio_data

    @property
    def fragment_shader_code(self):
        return self._fragment_shader_code

    @property
    def interpolation(self):
        return self._interpolation


@register_wgpu_render_function(gfx.WorldObject, AudioMaterial)
class AudioShader(BaseShader):
    # Mark as render-shader (as opposed to compute-shader)
    type = "render"

    def __init__(self, wobject, **kwargs):
        super().__init__(wobject, **kwargs)

        material = wobject.material

        fragment_shader_code = material.fragment_shader_code
        if fragment_shader_code:
            self["fragment_shader_code"] = fragment_shader_code

    def get_bindings(self, wobject, shared):
        material = wobject.material

        sampler = GfxSampler(material.interpolation, "clamp")
        view = GfxTextureView(material.audio_data, view_dim="2d")
        bindings = {
            0: Binding("s_data_map", "sampler/filtering", sampler, "FRAGMENT"),
            1: Binding("t_data_map", "texture/auto", view, "FRAGMENT"),
        }

        self.define_bindings(0, bindings)
        return {
            0: bindings,
        }

    def get_pipeline_info(self, wobject, shared):
        # We draw triangles, no culling
        return {
            "primitive_topology": wgpu.PrimitiveTopology.triangle_list,
            "cull_mode": wgpu.CullMode.none,
        }

    def get_render_info(self, wobject, shared):
        return {
            "indices": (3, 1),
            "render_mask": RenderMask.opaque,
        }

    def get_code(self):
        return """
        {{ bindings_code }}

        struct FragmentOutput {
            @location(0) color: vec4<f32>,
            @location(1) pick: vec4<u32>,
        };

        @vertex
        fn vs_main(@builtin(vertex_index) index: u32) -> Varyings {
            var varyings: Varyings;
            if (index == u32(0)) {
                varyings.position = vec4<f32>(-1.0, -1.0, 0.0, 1.0);
                varyings.uv = vec2<f32>(0.0, 1.0);
            } else if (index == u32(1)) {
                varyings.position = vec4<f32>(3.0, -1.0, 0.0, 1.0);
                varyings.uv = vec2<f32>(2.0, 1.0);
            } else {
                varyings.position = vec4<f32>(-1.0, 3.0, 0.0, 1.0);
                varyings.uv = vec2<f32>(0.0, -1.0);
            }
            return varyings;
        }

        $$ if fragment_shader_code is defined
        {{ fragment_shader_code }}
        $$ else

        @fragment
        fn fs_main(varyings: Varyings) -> FragmentOutput {
            var out: FragmentOutput;
            var uv = varyings.uv;
            uv.y = 1.0 - uv.y;
            let background_color = vec3<f32>(0.125, 0.125, 0.125);
            let color = vec3<f32>( 0.0, 1.0, 1.0 );
            let f = textureSample(t_data_map, s_data_map, vec2<f32>(uv.x, 0.0)).r;
            let i = step(uv.y, f) * step(f - 0.0125, uv.y);
            out.color = vec4<f32>(mix(background_color, color, i), 1.0);
            return out;
        }

        $$ endif
        """


################################################################################
# Demo starts here
################################################################################

fragment_shader_code1 = """
    fn srgb2physical(color: vec3<f32>) -> vec3<f32> {
        let f = pow((color + 0.055) / 1.055, vec3<f32>(2.4));
        let t = color / 12.92;
        return select(f, t, color <= vec3<f32>(0.04045));
    }

    @fragment
    fn fs_main(varyings: Varyings) -> FragmentOutput {
        var out: FragmentOutput;
        var uv = varyings.uv;
        uv.y = 1.0 - uv.y;

        let f = textureSample(t_data_map, s_data_map, vec2<f32>(abs(uv.x *2 -1.0), 0.0)).r;// led color

        // quantize coordinates
        let bands = 64.0;
        let segs = 40.0;
        var p = vec2<f32>(floor(uv.x*bands)/bands, floor(uv.y*segs)/segs);

        let color = mix(vec3<f32>(0.0, 2.0, 0.0), vec3<f32>(2.0, 0.0, 0.0), sqrt(uv.y));

        // mask for bar graph
        let mask = select(0.001, 1.0, p.y < f);

        let d = fract((uv - p) *vec2<f32>(bands, segs)) - 0.5;
        let led = smoothstep(0.5, 0.35, abs(d.x)) * smoothstep(0.5, 0.35, abs(d.y));

        // let led = step(d, 0.5 - gap);

        let ledColor = led*color*mask;
        out.color = vec4<f32>(srgb2physical(ledColor), 1.0);
        return out;
    }

"""

fragment_shader_code2 = """

    fn srgb2physical(color: vec3<f32>) -> vec3<f32> {
        let f = pow((color + 0.055) / 1.055, vec3<f32>(2.4));
        let t = color / 12.92;
        return select(f, t, color <= vec3<f32>(0.04045));
    }

    @fragment
    fn fs_main(varyings: Varyings) -> FragmentOutput {
        var out: FragmentOutput;
        var uv = varyings.uv;
        uv.y = 1.0 - uv.y;

        var p = uv*2.0-1.0;
        // p.x*=iResolution.x/iResolution.y;
        p.y+=0.5;

        var col = vec3f(0.0);
        var refs = vec3f(0.0);

        let nBands = 64.0;
        let i = floor(uv.x*nBands);
        let f = fract(uv.x*nBands);
        // var band = i/nBands;
        // band *= (band*band);
        // band = band*0.995;
        // band += 0.005;
        let s = textureSample(t_data_map, s_data_map, vec2f(uv.x, 0.0)).r;

        /* Gradient colors and amount here */
        let nColors = 4;
        var colors = array<vec3f, 4>(
            vec3f(0.0,0.0,1.0),
            vec3f(0.0,1.0,1.0),
            vec3f(1.0,1.0,0.0),
            vec3f(1.0,0.0,0.0)
        );

        var gradCol = colors[0];
        let n = f32(nColors)-1.0;
        for(var i = 1; i < nColors; i++)
        {
            var v = clamp((s-f32(i-1)/n)*n, 0.0, 1.0);
            gradCol = gradCol + v*(colors[i]-gradCol);
        }

        col += vec3f(1.0-smoothstep(0.0,0.01,p.y-s*1.5));
        col *= gradCol;

        refs += vec3f(1.0-smoothstep(0.0,-0.01,p.y+s*1.5));
        refs*= gradCol*smoothstep(-0.5,0.5,p.y);

        col = mix(refs,col,smoothstep(-0.01,0.01,p.y));

        col *= smoothstep(0.125,0.375,f);
        col *= smoothstep(0.875,0.625,f);

        col = clamp(col, vec3f(0.0), vec3f(1.0));

        out.color = vec4<f32>(srgb2physical(col), 1.0);
        return out;
    }

"""

fragment_shader_code3 = """

    fn srgb2physical(color: vec3<f32>) -> vec3<f32> {
        let f = pow((color + 0.055) / 1.055, vec3<f32>(2.4));
        let t = color / 12.92;
        return select(f, t, color <= vec3<f32>(0.04045));
    }

    fn light(d: f32, att: f32) -> f32 {
        return 1.0 / (1.0 + pow(abs(d * att), 1.3));
    }

    fn logX(x: f32, a: f32, c: f32) -> f32 {
        return 1.0 / (exp(-a * (x - c)) + 1.0);
    }

    fn getLevel(x: f32) -> f32 {
        return textureSample(t_data_map, s_data_map, vec2<f32>(x, 0.0)).r;
    }

    fn logisticAmp(amp: f32) -> f32 {
        let c = 0.88;
        let a = 20.0;
        return (logX(amp, a, c) - logX(0.0, a, c)) / (logX(1.0, a, c) - logX(0.0, a, c));
    }

    fn getPitch(freq: ptr<function, f32>, octave: f32) -> f32 {
        *freq = pow(2., *freq)  * 261.;
        *freq = pow(2., octave) * *freq / 12000.;
        return logisticAmp(getLevel(*freq));
    }
    fn getVol(samples: f32) -> f32{
        var avg = 0.;
        for (var i = 0.; i < samples; i=i+1) {
            avg += getLevel(i/samples);
        }
        return avg / samples;
    }

    fn sdBox(p: vec3<f32>, b: vec3<f32>) -> f32 {
        let q = abs(p) - b;
        return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
    }

    fn hash13( p3: ptr<function, vec3f> ) -> f32 {
        *p3  = fract(*p3 * .1031);
        *p3 += dot(*p3, (*p3).zyx + 31.32);
        return fract(((*p3).x + (*p3).y) * (*p3).z);
    }

    @fragment
    fn fs_main(varyings: Varyings) -> FragmentOutput {
        var out: FragmentOutput;
        var uv = varyings.uv;
        uv.y = 1.0 - uv.y;

        uv = uv * 2.0 - 1.0;

        uv.x = uv.x * 2.0;

        var col = vec3f(.1,.0,.14);
        let vol = getVol(8.);

        let ro = vec3f(0, 8, 12)*(1. + vol*.3);
        // ro.zx *= rot(iTime*.4);
        let f = normalize(-ro);
        let r = normalize(cross(vec3<f32>(0.0, 1.0, 0.0), f));
        let rd = normalize(f + uv.x*r + uv.y*cross(f, r));

        var t = 0.0;
        for (var i = 0.; i < 30.; i += 1.0) {
            let p  = ro + t*rd;

            let cen = floor(p.xz) + .5;
            var id = abs(vec3(cen.x, 0, cen.y));
            let d = length(id);

            var freq = smoothstep(0., 20., d)*3 + hash13(&id)*2.;
            let pitch = getPitch(&freq, .7);

            let v  = vol*smoothstep(2., 0., d);
            let h  = d*.2*(1.+pitch*1.5) + v*2.;
            let me = sdBox(p - vec3f(cen.x, -50., cen.y), vec3f(.3, 50. + h, .3)+pitch) - .05;

            col += mix(
                mix(vec3<f32>(0.8, 0.2, 0.4), vec3<f32>(0.0, 1.0, 0.0), min(v * 2.0, 1.0)),
                vec3<f32>(0.5, 0.3, 1.2),
                smoothstep(10.0, 30.0, d)
            ) * (cos(id) + 1.5) * (pitch * d * 0.08 + v) * light(me, 20.0) * (1.0 + vol * 2.0);

            t += me;
        }

        out.color = vec4<f32>(srgb2physical(col), 1.0);
        return out;
    }

"""

fragment_shader_code4 = """

    fn srgb2physical(color: vec3<f32>) -> vec3<f32> {
        let f = pow((color + 0.055) / 1.055, vec3<f32>(2.4));
        let t = color / 12.92;
        return select(f, t, color <= vec3<f32>(0.04045));
    }

    fn getAmp(frequency: f32) -> f32{
        return textureSample(t_data_map, s_data_map, vec2f(frequency / 512.0, 0.0)).r;
    }

    fn getWeight(f: f32) -> f32{
        return (getAmp(f-2.0) + getAmp(f-1.0) + getAmp(f+2.0) + getAmp(f+1.0) + getAmp(f)) / 5.0; }

    @fragment
    fn fs_main(varyings: Varyings) -> FragmentOutput {
        var out: FragmentOutput;
        var uv = varyings.uv;
        uv.y = 1.0 - uv.y;

        let uv_raw = uv;

        uv = -1.0 + 2.0 * uv;

        var lineIntensity: f32;
        var glowWidth: f32;
        var color = vec3f(0.0);

        for(var i = 0.0; i < 5.0; i=i+1.0) {

            uv.y += (0.3 * sin(uv.y + i - 5.0 ));
            let Y = uv.y + getWeight(i*i*20.0) * (textureSample(t_data_map, s_data_map, vec2f(uv_raw.x, 0.0)).r - 0.5);
            let s = 0.6 * abs( (uv.x + i / 4.3) % 2.0 - 1.0);
            lineIntensity = 0.5 + s * s;
            glowWidth = abs(lineIntensity / (150.0 * Y));
            color += vec3f(glowWidth * (1.5 ), glowWidth * (1.5 ), glowWidth * (0.5 ));
        }

        out.color = vec4<f32>(srgb2physical(color), 1.0);
        return out;
    }


"""

# Setup scene

FFT_SIZE = 128

renderer = gfx.WgpuRenderer(
    WgpuCanvas(title="audio visualizer", max_fps=60, size=(1280, 720))
)
camera = gfx.NDCCamera()  # Not actually used

t_audio_freq = gfx.Texture(
    np.zeros((FFT_SIZE // 2, 1), dtype=np.uint8),
    size=(FFT_SIZE // 2, 1, 1),
    format="r8unorm",
    dim=2,
)

vp1 = gfx.Viewport(renderer, (0, 0, 640, 360))
scene1 = gfx.Scene()
wo = gfx.WorldObject(None, AudioMaterial(t_audio_freq, fragment_shader_code=None))
scene1.add(wo)

vp2 = gfx.Viewport(renderer, (0, 360, 640, 360))
scene2 = gfx.Scene()
wo2 = gfx.WorldObject(
    None, AudioMaterial(t_audio_freq, fragment_shader_code=fragment_shader_code2)
)
scene2.add(wo2)

vp3 = gfx.Viewport(renderer, (640, 0, 640, 360))
scene3 = gfx.Scene()
wo3 = gfx.WorldObject(
    None, AudioMaterial(t_audio_freq, fragment_shader_code=fragment_shader_code3)
)
scene3.add(wo3)

t_audio_time_domain = gfx.Texture(
    np.zeros((FFT_SIZE, 1), dtype=np.uint8),
    size=(FFT_SIZE, 1, 1),
    format="r8unorm",
    dim=2,
)

vp4 = gfx.Viewport(renderer, (640, 360, 640, 360))
scene4 = gfx.Scene()
wo4 = gfx.WorldObject(
    None,
    AudioMaterial(
        t_audio_time_domain,
        interpolation="linear",
        fragment_shader_code=fragment_shader_code4,
    ),
)
scene4.add(wo4)

analyzer = AudioAnalyzer(FFT_SIZE)
audio_player = AudioPlayer()
audio_player.analyzer = analyzer


def animate():
    t_audio_freq.data.flat = analyzer.get_byte_frequency_data()
    t_audio_freq.update_range((0, 0, 0), (FFT_SIZE // 2, 1, 1))

    t_audio_time_domain.data.flat = analyzer.get_byte_time_domain_data()
    t_audio_time_domain.update_range((0, 0, 0), (FFT_SIZE, 1, 1))
    vp1.render(scene1, camera)
    vp2.render(scene2, camera)
    vp3.render(scene3, camera)
    vp4.render(scene4, camera)
    renderer.flush()
    renderer.request_draw()


if __name__ == "__main__":
    song_path = "https://audio-download.ngfiles.com/376000/376737_Skullbeatz___Bad_Cat_Maste.mp3"
    audio_player.play(song_path)
    renderer.request_draw(animate)
    run()
