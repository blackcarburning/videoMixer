

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import time
import os
import json
try:
    import winsound
except ImportError:
    winsound = None
import queue
import random
import ctypes
import pygame
import pygame.sndarray

# --- PYGAME AUDIO ENGINE ---
class NativeAudioEngine:
    def __init__(self):
        # Initialize pygame.mixer only if not already initialized
        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=1024)
        self.is_loaded = False
        self.duration_ms = 0
        self.is_paused = False
        self.start_time_ms = 0
        self.pause_time_ms = 0
        
    def load(self, path):
        self.stop()
        self.close()
        try:
            pygame.mixer.music.load(path)
            self.is_loaded = True
            # Get duration if possible using pygame
            try:
                sound = pygame.mixer.Sound(path)
                self.duration_ms = int(sound.get_length() * 1000)
            except:
                self.duration_ms = 0
            return True
        except Exception as e:
            print(f"Failed to load audio: {e}")
            self.is_loaded = False
            return False

    def play(self, from_ms=0):
        if not self.is_loaded:
            return
        try:
            pygame.mixer.music.play(loops=0, start=from_ms / 1000.0)
            self.start_time_ms = from_ms
            self.is_paused = False
        except Exception as e:
            print(f"Failed to play audio: {e}")

    def is_playing(self):
        if not self.is_loaded:
            return False
        return pygame.mixer.music.get_busy() and not self.is_paused

    def get_position(self):
        if not self.is_loaded:
            return 0
        if self.is_paused:
            return self.pause_time_ms
        try:
            pos_sec = pygame.mixer.music.get_pos() / 1000.0
            return int(self.start_time_ms + pos_sec * 1000)
        except:
            return 0

    def pause(self):
        if self.is_loaded and not self.is_paused:
            # Calculate current position before pausing
            try:
                pos_sec = pygame.mixer.music.get_pos() / 1000.0
                self.pause_time_ms = int(self.start_time_ms + pos_sec * 1000)
            except:
                self.pause_time_ms = 0
            pygame.mixer.music.pause()
            self.is_paused = True

    def unpause(self):
        if self.is_loaded and self.is_paused:
            pygame.mixer.music.unpause()
            self.is_paused = False

    def stop(self):
        if self.is_loaded:
            pygame.mixer.music.stop()
            self.is_paused = False
            self.start_time_ms = 0

    def close(self):
        if self.is_loaded:
            pygame.mixer.music.unload()
        self.is_loaded = False
        self.is_paused = False

class AudioChannel:
    def __init__(self):
        self.engine = NativeAudioEngine()
        self.enabled = True
        self.path = None
        self.playing = False
        
    def load(self, path):
        if os.path.exists(path):
            self.path = path
            return self.engine.load(path)
        return False
            
    def play(self, start_time_ms=0):
        if self.enabled and self.engine.is_loaded:
            self.engine.play(start_time_ms)
            self.playing = True
            
    def get_time_ms(self):
        return self.engine.get_position()
            
    def is_active(self):
        return self.engine.is_playing()

    def stop(self):
        self.engine.stop()
        self.playing = False
        
    def close(self):
        self.engine.close()
        self.playing = False

class HighPrecisionMetronome:
    def __init__(self):
        self.enabled = False
        self.volume = 0.5
        self.high_freq = 1200
        self.low_freq = 800
        self.duration = 40
        self.bpm = 120.0
        self.beats_per_bar = 4
        self.lock = threading.Lock()
        self.current_beat = -1.0
        
    def update_volume(self, volume):
        with self.lock:
            self.volume = volume
            self.duration = int(30 + volume * 30)
    
    def set_bpm(self, bpm):
        with self.lock:
            self.bpm = bpm
            
    def set_beats_per_bar(self, beats):
        with self.lock:
            self.beats_per_bar = beats
            
    def prepare_start(self):
        self.current_beat = -1.0
        
    def synchronized_start(self, start_time):
        self.current_beat = -1.0
        
    def stop(self):
        self.current_beat = -1.0

    def play_click(self, is_downbeat=False):
        try:
            if self.enabled and winsound is not None:
                winsound.Beep(self.high_freq if is_downbeat else self.low_freq, self.duration)
        except:
            pass
    
    def check_tick(self, beat_pos):
        if not self.enabled: return
        cb = int(beat_pos)
        if cb > int(self.current_beat):
            is_down = (cb % self.beats_per_bar) == 0
            threading.Thread(target=self.play_click, args=(is_down,), daemon=True).start()
        self.current_beat = beat_pos

def generate_snare_sound():
    """Generate a simple snare drum sound using white noise and sine wave"""
    sample_rate = 44100
    duration = 0.15  # 150ms
    samples = int(sample_rate * duration)
    
    # Create white noise for the snare body
    noise = np.random.uniform(-1, 1, samples).astype(np.float32)
    
    # Add a short sine wave for the "snap"
    tone_freq = 180
    t = np.linspace(0, duration, samples, dtype=np.float32)
    tone = np.sin(2 * np.pi * tone_freq * t).astype(np.float32)
    
    # Mix noise and tone
    mix = 0.7 * noise + 0.3 * tone
    
    # Apply envelope (quick attack, exponential decay)
    envelope = np.exp(-t * 15).astype(np.float32)
    snare = mix * envelope
    
    # Normalize and convert to 16-bit
    snare = snare / np.max(np.abs(snare)) * 0.6
    snare_16bit = (snare * 32767).astype(np.int16)
    
    # Create stereo sound
    stereo = np.column_stack((snare_16bit, snare_16bit))
    
    return pygame.sndarray.make_sound(stereo)

# Pre-generate snare sound
SNARE_SOUND = None
try:
    SNARE_SOUND = generate_snare_sound()
except Exception as e:
    print(f"Failed to generate snare sound: {e}")

class Modulator:
    RATE_OPTIONS = {"1/32": 0.03125, "1/16": 0.0625, "1/8": 0.125, "1/4": 0.25, "1/2": 0.5,
                    "1": 1.0, "2": 2.0, "4": 4.0, "8": 8.0, "16": 16.0, "32": 32.0}
    RATE_REVERSE = {v: k for k, v in RATE_OPTIONS.items()}
    _sin_table = np.sin(np.linspace(0, 2 * np.pi, 4096, dtype=np.float32))
    WAVE_TYPES = ["sine", "square", "triangle", "saw_forward", "saw_backward", "envelope"]

    def __init__(self):
        self.wave_type = "sine"
        self.rate = 1.0
        self.depth = 1.0
        self.phase = 0.0
        self.enabled = False
        self.pos_only = False
        self.neg_only = False
        self.invert = False
        self.fwd_only = False
        self.rev_only = False
        self.fine_tune = 0.0
        
    def get_value(self, beat_position):
        if not self.enabled or self.depth == 0:
            return 0.0
        cp = ((beat_position / self.rate) + self.phase) % 1.0
        value = 0.0
        if self.wave_type == "sine":
            value = Modulator._sin_table[int(cp * 4095)]
        elif self.wave_type == "square":
            value = 1.0 if cp < 0.5 else -1.0
        elif self.wave_type == "triangle":
            value = 4*cp if cp < 0.25 else (2-4*cp if cp < 0.75 else 4*cp-4)
        elif self.wave_type == "saw_forward":
            value = 2.0 * cp - 1.0
        elif self.wave_type == "saw_backward":
            value = 1.0 - 2.0 * cp
        elif self.wave_type == "envelope":
            value = pow(1.0 - cp, 4) * 2.0 - 1.0 
            if value < -1.0: value = -1.0
        if self.invert: value = -value
        if self.pos_only: value = max(0, value)
        elif self.neg_only: value = min(0, value)
        
        # Apply direction limiters (for speed modulator)
        if self.fwd_only:
            value = max(0, value)
        elif self.rev_only:
            value = min(0, value)
        
        return value * self.depth
    
    def to_dict(self):
        return {'wave_type': self.wave_type, 'rate': self.rate, 'depth': self.depth, 
                'phase': self.phase, 'enabled': self.enabled, 'pos_only': self.pos_only, 
                'neg_only': self.neg_only, 'invert': self.invert, 'fwd_only': self.fwd_only, 
                'rev_only': self.rev_only, 'fine_tune': self.fine_tune}
    
    def from_dict(self, d):
        self.wave_type = d.get('wave_type', 'sine')
        self.rate = d.get('rate', 1.0)
        self.depth = d.get('depth', 1.0)
        self.phase = d.get('phase', 0.0)
        self.enabled = d.get('enabled', False)
        self.pos_only = d.get('pos_only', False)
        self.neg_only = d.get('neg_only', False)
        self.invert = d.get('invert', False)
        self.fwd_only = d.get('fwd_only', False)
        self.rev_only = d.get('rev_only', False)
        self.fine_tune = d.get('fine_tune', 0.0)
    
    def reset(self):
        self.wave_type = "sine"
        self.rate = 1.0
        self.depth = 1.0
        self.phase = 0.0
        self.enabled = False
        self.pos_only = False
        self.neg_only = False
        self.invert = False
        self.fwd_only = False
        self.rev_only = False
        self.fine_tune = 0.0

class VideoChannel:
    SPEED_OPTIONS = {"1/6": 1/6, "1/4": 0.25, "1/3": 1/3, "1/2": 0.5, "2/3": 2/3, "3/4": 0.75,
                     "1": 1.0, "1.5": 1.5, "2": 2.0, "3": 3.0, "4": 4.0, "6": 6.0, "8": 8.0, "12": 12.0, "16": 16.0}
    SPEED_LABELS = list(SPEED_OPTIONS.keys())
    LOOP_LENGTH_OPTIONS = {
        "1/32 bt": 0.03125, "1/16 bt": 0.0625, "1/8 bt": 0.125, "1/4 bt": 0.25, 
        "1/2 bt": 0.5, "1 bt": 1.0, "2 bt": 2.0, "1 bar": 4.0, "2 bar": 8.0, 
        "4 bar": 16.0, "8 bar": 32.0
    }
    LOOP_LENGTH_LABELS = list(LOOP_LENGTH_OPTIONS.keys())
    STROBE_RATES = {"1/4": 0.25, "1/8": 0.125, "1/16": 0.0625, "1/32": 0.03125}
    POSTERIZE_RATES = {"Off": 0.0, "1/16": 0.0625, "1/8": 0.125, "1/4": 0.25, "1/2": 0.5, "1 bt": 1.0}
    MIRROR_MODES = ["Off", "Horizontal", "Vertical", "Quad", "Kaleido"]
    
    # Gate sequencer constants
    DEFAULT_BPM = 120.0
    MIN_ENVELOPE_TIME = 0.01
    DEFAULT_ENVELOPE_TIME = 0.05
    
    def __init__(self, target_width, target_height):
        self.video_path = None
        self.cap = None
        self.frame_count = 0
        self.fps = 30
        self.width = 0
        self.height = 0
        self.brightness = 0.0
        self.contrast = 1.0
        self.saturation = 1.0
        self.speed = 1.0
        self.opacity = 1.0
        self.reverse = False
        self.glitch_rate = 0.0
        self.strobe_enabled = False
        self.strobe_rate = 0.125
        self.strobe_color = "white"
        self.posterize_rate = 0.0
        self.mirror_mode = "Off"
        self.mosh_amount = 0.0
        self.mosh_buffer = None
        self.seq_gate = [1] * 16
        self.seq_stutter = [0] * 16
        self.seq_speed = [0] * 16
        self.seq_jump = [0] * 16
        self.gate_snare_enabled = False
        self.gate_envelope_enabled = False
        self.gate_attack = 0.0
        self.gate_decay = 0.0
        self.gate_timebase = 4.0
        self.brightness_mod = Modulator()
        self.contrast_mod = Modulator()
        self.saturation_mod = Modulator()
        self.opacity_mod = Modulator()
        self.loop_start_mod = Modulator()
        self.rgb_mod = Modulator()
        self.blur_mod = Modulator()
        self.zoom_mod = Modulator()
        self.pixel_mod = Modulator()
        self.chroma_mod = Modulator()
        self.chroma_mod.wave_type = "sine"
        self.chroma_mod.rate = 1.0
        self.chroma_mod.depth = 0.5
        self.mirror_center_mod = Modulator()
        self.mirror_center_mod.wave_type = "triangle"
        self.mirror_center_mod.rate = 0.5
        self.mirror_center_mod.depth = 0.3
        self.speed_mod = Modulator()
        self.speed_mod.wave_type = "sine"
        self.speed_mod.rate = 2.0
        self.speed_mod.depth = 1.0
        self.mosh_mod = Modulator()
        self.mosh_mod.wave_type = "sine"
        self.mosh_mod.rate = 0.25
        self.mosh_mod.depth = 1.0
        self.echo_amount = 0.0
        self.echo_buffer = None
        self.echo_mod = Modulator()
        self.echo_mod.wave_type = "triangle"
        self.echo_mod.rate = 1.0
        self.echo_mod.depth = 1.0
        self.slicer_amount = 0.0
        self.slicer_buffer = None
        self.slicer_mod = Modulator()
        self.slicer_mod.wave_type = "square"
        self.slicer_mod.rate = 2.0
        self.slicer_mod.depth = 1.0
        self.kaleidoscope_amount = 0.0
        self.kaleidoscope_buffer = None
        self.kaleidoscope_mod = Modulator()
        self.kaleidoscope_mod.wave_type = "sine"
        self.kaleidoscope_mod.rate = 0.5
        self.kaleidoscope_mod.depth = 1.0
        self.vignette_amount = 0.0
        self.vignette_transparency = 0.5
        self.vignette_mod = Modulator()
        self.vignette_mod.wave_type = "sine"
        self.vignette_mod.rate = 0.25
        self.vignette_mod.depth = 1.0
        self.color_shift_amount = 0.0
        self.color_shift_mod = Modulator()
        self.color_shift_mod.wave_type = "triangle"
        self.color_shift_mod.rate = 1.0
        self.color_shift_mod.depth = 1.0
        self.spin_amount = 0.0
        self.spin_mod = Modulator()
        self.spin_mod.wave_type = "sine"
        self.spin_mod.rate = 1.0
        self.spin_mod.depth = 1.0
        self.loop = True
        self.playback_position = 0.0
        self.beat_loop_enabled = False
        self.loop_length_beats = 4.0 
        self.loop_start_frame = 0
        self.lock = threading.RLock()
        self.target_width = target_width
        self.target_height = target_height
        self.resized_cache = {}
        self.cache_size = 120
        self.current_frame_idx = -1
        self.current_resized = None
        self.last_posterize_beat = -1.0
        self.last_gate_step = -1
    
    def set_target_size(self, w, h):
        if w != self.target_width or h != self.target_height:
            self.target_width = w
            self.target_height = h
            self.resized_cache.clear()
            self.current_resized = None
            self.mosh_buffer = None
            self.echo_buffer = None
            self.slicer_buffer = None
            self.kaleidoscope_buffer = None
    
    def _get_gate_step(self, beat_pos):
        """Calculate the current gate sequencer step based on beat position and timebase"""
        return int((beat_pos % self.gate_timebase) * (16.0 / self.gate_timebase)) % 16
    
    def _get_step_timing(self, bpm):
        """Calculate step duration and position for envelope calculations"""
        step_duration_beats = self.gate_timebase / 16.0
        beat_duration_sec = 60.0 / (bpm if bpm > 0 else self.DEFAULT_BPM)
        step_duration_sec = step_duration_beats * beat_duration_sec
        return step_duration_sec
    
    def to_dict(self, include_video=False):
        d = {'brightness': self.brightness, 'contrast': self.contrast, 'saturation': self.saturation,
             'speed': self.speed, 'opacity': self.opacity, 'loop': self.loop,
             'reverse': self.reverse, 'glitch_rate': self.glitch_rate,
             'strobe_enabled': self.strobe_enabled, 'strobe_rate': self.strobe_rate, 'strobe_color': self.strobe_color,
             'posterize_rate': self.posterize_rate, 'mirror_mode': self.mirror_mode, 'mosh_amount': self.mosh_amount,
             'echo_amount': self.echo_amount, 'slicer_amount': self.slicer_amount,
             'kaleidoscope_amount': self.kaleidoscope_amount, 'vignette_amount': self.vignette_amount,
             'vignette_transparency': self.vignette_transparency, 'color_shift_amount': self.color_shift_amount,
             'seq_gate': self.seq_gate, 'seq_stutter': self.seq_stutter, 
             'seq_speed': self.seq_speed, 'seq_jump': self.seq_jump,
             'gate_snare_enabled': self.gate_snare_enabled, 'gate_envelope_enabled': self.gate_envelope_enabled,
             'gate_attack': self.gate_attack, 'gate_decay': self.gate_decay, 'gate_timebase': self.gate_timebase,
             'beat_loop_enabled': self.beat_loop_enabled, 
             'loop_length_beats': self.loop_length_beats,
             'loop_start_frame': self.loop_start_frame,
             'brightness_mod': self.brightness_mod.to_dict(), 'contrast_mod': self.contrast_mod.to_dict(),
             'saturation_mod': self.saturation_mod.to_dict(), 'opacity_mod': self.opacity_mod.to_dict(),
             'loop_start_mod': self.loop_start_mod.to_dict(), 'rgb_mod': self.rgb_mod.to_dict(),
             'blur_mod': self.blur_mod.to_dict(), 'zoom_mod': self.zoom_mod.to_dict(), 'pixel_mod': self.pixel_mod.to_dict(),
             'chroma_mod': self.chroma_mod.to_dict(), 'mosh_mod': self.mosh_mod.to_dict(),
             'echo_mod': self.echo_mod.to_dict(), 'slicer_mod': self.slicer_mod.to_dict(),
             'mirror_center_mod': self.mirror_center_mod.to_dict(), 'speed_mod': self.speed_mod.to_dict(),
             'kaleidoscope_mod': self.kaleidoscope_mod.to_dict(), 'vignette_mod': self.vignette_mod.to_dict(),
             'color_shift_mod': self.color_shift_mod.to_dict(), 'spin_amount': self.spin_amount,
             'spin_mod': self.spin_mod.to_dict()}
        if include_video:
            d['video_path'] = self.video_path
        return d
    
    def from_dict(self, d, load_video=False):
        with self.lock:
            self.brightness = d.get('brightness', 0.0)
            self.contrast = d.get('contrast', 1.0)
            self.saturation = d.get('saturation', 1.0)
            self.speed = d.get('speed', 1.0)
            self.opacity = d.get('opacity', 1.0)
            self.loop = d.get('loop', True)
            self.reverse = d.get('reverse', False)
            self.glitch_rate = d.get('glitch_rate', 0.0)
            self.strobe_enabled = d.get('strobe_enabled', False)
            self.strobe_rate = d.get('strobe_rate', 0.125)
            self.strobe_color = d.get('strobe_color', 'white')
            self.posterize_rate = d.get('posterize_rate', 0.0)
            self.mirror_mode = d.get('mirror_mode', "Off")
            self.mosh_amount = d.get('mosh_amount', 0.0)
            self.echo_amount = d.get('echo_amount', 0.0)
            self.slicer_amount = d.get('slicer_amount', 0.0)
            self.kaleidoscope_amount = d.get('kaleidoscope_amount', 0.0)
            self.vignette_amount = d.get('vignette_amount', 0.0)
            self.vignette_transparency = d.get('vignette_transparency', 0.5)
            self.color_shift_amount = d.get('color_shift_amount', 0.0)
            self.spin_amount = d.get('spin_amount', 0.0)
            self.seq_gate = d.get('seq_gate', [1]*16)
            self.seq_stutter = d.get('seq_stutter', [0]*16)
            self.seq_speed = d.get('seq_speed', [0]*16)
            self.seq_jump = d.get('seq_jump', [0]*16)
            self.gate_snare_enabled = d.get('gate_snare_enabled', False)
            self.gate_envelope_enabled = d.get('gate_envelope_enabled', False)
            self.gate_attack = d.get('gate_attack', 0.0)
            self.gate_decay = d.get('gate_decay', 0.0)
            self.gate_timebase = d.get('gate_timebase', 4.0)
            self.beat_loop_enabled = d.get('beat_loop_enabled', False)
            self.loop_length_beats = d.get('loop_length_beats', 4.0)
            self.loop_start_frame = d.get('loop_start_frame', 0)
            for m in ['brightness_mod', 'contrast_mod', 'saturation_mod', 'opacity_mod', 'loop_start_mod', 'rgb_mod', 'blur_mod', 'zoom_mod', 'pixel_mod', 'chroma_mod', 'mosh_mod', 'echo_mod', 'slicer_mod', 'mirror_center_mod', 'speed_mod', 'kaleidoscope_mod', 'vignette_mod', 'color_shift_mod', 'spin_mod']:
                if m in d:
                    getattr(self, m).from_dict(d[m])
            if load_video and d.get('video_path'):
                self.load_video(d['video_path'])
        
    def load_video(self, path):
        with self.lock:
            if self.cap:
                self.cap.release()
            self.cap = cv2.VideoCapture(path)
            if not self.cap.isOpened():
                return False
            self.video_path = path
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.playback_position = 0.0
            self.resized_cache.clear()
            self.current_frame_idx = -1
            self.current_resized = None
            self.mosh_buffer = None
            self.echo_buffer = None
            self.slicer_buffer = None
            return True
    
    def reset_position(self):
        with self.lock:
            if self.cap:
                st = self.loop_start_frame if self.beat_loop_enabled else 0
                self.playback_position = float(st)
                self.current_frame_idx = -1
    
    def reset_controls(self):
        with self.lock:
            self.brightness = 0.0
            self.contrast = 1.0
            self.saturation = 1.0
            self.speed = 1.0
            self.opacity = 1.0
            self.loop = True
            self.reverse = False
            self.glitch_rate = 0.0
            self.strobe_enabled = False
            self.strobe_rate = 0.125
            self.strobe_color = "white"
            self.posterize_rate = 0.0
            self.mirror_mode = "Off"
            self.mosh_amount = 0.0
            self.echo_amount = 0.0
            self.slicer_amount = 0.0
            self.kaleidoscope_amount = 0.0
            self.vignette_amount = 0.0
            self.vignette_transparency = 0.5
            self.color_shift_amount = 0.0
            self.seq_gate = [1] * 16
            self.seq_stutter = [0] * 16
            self.seq_speed = [0] * 16
            self.seq_jump = [0] * 16
            self.gate_snare_enabled = False
            self.gate_envelope_enabled = False
            self.gate_attack = 0.0
            self.gate_decay = 0.0
            self.gate_timebase = 4.0
            self.beat_loop_enabled = False
            self.loop_length_beats = 4.0
            self.loop_start_frame = 0
            self.brightness_mod.reset()
            self.contrast_mod.reset()
            self.saturation_mod.reset()
            self.opacity_mod.reset()
            self.loop_start_mod.reset()
            self.rgb_mod.reset()
            self.blur_mod.reset()
            self.zoom_mod.reset()
            self.pixel_mod.reset()
            self.chroma_mod.reset()
            self.chroma_mod.wave_type = "sine"
            self.chroma_mod.rate = 1.0
            self.chroma_mod.depth = 0.5
            self.mirror_center_mod.reset()
            self.mirror_center_mod.wave_type = "triangle"
            self.mirror_center_mod.rate = 0.5
            self.mirror_center_mod.depth = 0.3
            self.speed_mod.reset()
            self.speed_mod.wave_type = "sine"
            self.speed_mod.rate = 2.0
            self.speed_mod.depth = 1.0
            self.mosh_mod.reset()
            self.mosh_mod.wave_type = "sine"
            self.mosh_mod.rate = 0.25
            self.mosh_mod.depth = 1.0
            self.echo_mod.reset()
            self.echo_mod.wave_type = "triangle"
            self.echo_mod.rate = 1.0
            self.echo_mod.depth = 1.0
            self.slicer_mod.reset()
            self.slicer_mod.wave_type = "square"
            self.slicer_mod.rate = 2.0
            self.slicer_mod.depth = 1.0
            self.kaleidoscope_mod.reset()
            self.kaleidoscope_mod.wave_type = "sine"
            self.kaleidoscope_mod.rate = 0.5
            self.kaleidoscope_mod.depth = 1.0
            self.vignette_mod.reset()
            self.vignette_mod.wave_type = "sine"
            self.vignette_mod.rate = 0.25
            self.vignette_mod.depth = 1.0
            self.color_shift_mod.reset()
            self.color_shift_mod.wave_type = "triangle"
            self.color_shift_mod.rate = 1.0
            self.color_shift_mod.depth = 1.0
            self.spin_amount = 0.0
            self.spin_mod.reset()
            self.spin_mod.wave_type = "sine"
            self.spin_mod.rate = 1.0
            self.spin_mod.depth = 1.0
    
    def _get_resized(self, frame_idx, raw_frame):
        if frame_idx in self.resized_cache:
            return self.resized_cache[frame_idx]
        resized = cv2.resize(raw_frame, (self.target_width, self.target_height), interpolation=cv2.INTER_LINEAR)
        resized = resized.astype(np.float32) * (1.0/255.0)
        if len(self.resized_cache) >= self.cache_size:
            keys = sorted(self.resized_cache.keys())
            for k in keys[:20]:
                del self.resized_cache[k]
        self.resized_cache[frame_idx] = resized
        return resized
    
    def _apply_mirror(self, frame, beat_pos):
        if self.mirror_mode == "Off":
            return frame
        
        # Get modulator value (-depth to +depth) and map to 0.2-0.8 range
        mod_value = self.mirror_center_mod.get_value(beat_pos) if self.mirror_center_mod.enabled else 0.0
        # Map from [-depth, +depth] to [0.2, 0.8]
        center_offset = 0.5 + mod_value * 0.3  # Maps to 0.2-0.8 range with depth=0.3
        # Clamp to 0.2 - 0.8 range
        center_offset = max(0.2, min(0.8, center_offset))
        
        h, w = frame.shape[:2]
        
        if self.mirror_mode == "Horizontal":
            split_x = int(w * center_offset)
            left = frame[:, :split_x]
            # Resize left portion to half width, then mirror
            left_resized = cv2.resize(left, (w//2, h), interpolation=cv2.INTER_LINEAR)
            return np.hstack([left_resized, cv2.flip(left_resized, 1)])
        elif self.mirror_mode == "Vertical":
            split_y = int(h * center_offset)
            top = frame[:split_y, :]
            # Resize top portion to half height, then mirror
            top_resized = cv2.resize(top, (w, h//2), interpolation=cv2.INTER_LINEAR)
            return np.vstack([top_resized, cv2.flip(top_resized, 0)])
        elif self.mirror_mode == "Quad":
            split_y = int(h * center_offset)
            split_x = int(w * center_offset)
            q = frame[:split_y, :split_x]
            # Resize to quarter size
            q_resized = cv2.resize(q, (w//2, h//2), interpolation=cv2.INTER_LINEAR)
            top = np.hstack([q_resized, cv2.flip(q_resized, 1)])
            return np.vstack([top, cv2.flip(top, 0)])
        elif self.mirror_mode == "Kaleido":
            split_y = int(h * center_offset)
            split_x = int(w * center_offset)
            q = frame[:split_y, :split_x]
            q_resized = cv2.resize(q, (w//2, h//2), interpolation=cv2.INTER_LINEAR)
            q_flip = cv2.flip(q_resized, 1)
            top = np.hstack([q_resized, q_flip])
            bot = cv2.flip(top, 0)
            return np.vstack([top, bot])
        return frame

    def get_frame(self, beat_pos, delta_time, bpm, bpb):
        if not self.cap or self.frame_count == 0:
            return None
            
        with self.lock:
            target_idx = -1
            should_update = True
            if self.posterize_rate > 0:
                current_slot = int(beat_pos / self.posterize_rate)
                if current_slot == self.last_posterize_beat: should_update = False
                else: self.last_posterize_beat = current_slot
            
            if not should_update and self.current_resized is not None:
                frame = self.current_resized
            else:
                seq_step = int((beat_pos % 4.0) * 4) % 16
                spd_mod_idx = self.seq_speed[seq_step]
                seq_speed_mult = 1.0
                if spd_mod_idx == 1: seq_speed_mult = 2.0
                elif spd_mod_idx == 2: seq_speed_mult = 0.5
                elif spd_mod_idx == 3: seq_speed_mult = -1.0
                elif spd_mod_idx == 4: seq_speed_mult = 0.0
                
                scaled_beat_pos = beat_pos * self.speed
                jmp_mod_idx = self.seq_jump[seq_step]
                jump_offset_beats = 0.0
                if jmp_mod_idx == 1: jump_offset_beats = -1.0
                elif jmp_mod_idx == 2: jump_offset_beats = -4.0
                
                eff_beat_pos = scaled_beat_pos + (jump_offset_beats * self.speed)
                is_stuttering = self.seq_stutter[seq_step]
                
                if is_stuttering or spd_mod_idx == 4: 
                    quantized_beat = int(eff_beat_pos * 4) / 4.0
                    if self.beat_loop_enabled:
                         loop_beats = self.loop_length_beats
                         if loop_beats <= 0: loop_beats = 1.0
                         prog = (quantized_beat % loop_beats) / loop_beats
                         if self.reverse: prog = 1.0 - prog
                         loop_frames = int(loop_beats * (60.0/bpm) * self.fps)
                         base_start = self.loop_start_frame
                         mod_offset = 0
                         if self.loop_start_mod.enabled:
                             mod_offset = int(self.loop_start_mod.get_value(beat_pos) * self.frame_count)
                         fine_tune_offset = int((self.loop_start_mod.fine_tune / 100.0) * 0.01 * self.frame_count)
                         target_idx = (base_start + mod_offset + fine_tune_offset + int(prog * loop_frames)) % self.frame_count
                    else:
                         seconds = quantized_beat * (60.0 / bpm)
                         target_idx = int(seconds * self.fps) % self.frame_count
                
                elif self.beat_loop_enabled:
                    loop_beats = self.loop_length_beats
                    if loop_beats <= 0: loop_beats = 1.0
                    prog = (eff_beat_pos % loop_beats) / loop_beats
                    eff_rev = self.reverse
                    if spd_mod_idx == 3: eff_rev = not eff_rev
                    if eff_rev: prog = 1.0 - prog
                    loop_frames = int(loop_beats * (60.0 / bpm) * self.fps)
                    base_start = self.loop_start_frame
                    mod_offset = 0
                    if self.loop_start_mod.enabled:
                        mod_offset = int(self.loop_start_mod.get_value(beat_pos) * self.frame_count)
                    fine_tune_offset = int((self.loop_start_mod.fine_tune / 100.0) * 0.01 * self.frame_count)
                    target_idx = (base_start + mod_offset + fine_tune_offset + int(prog * loop_frames)) % self.frame_count
                
                else:
                    # Apply speed modulator: modulator range [-1, 1] maps to speed multiplier [-8, 8]
                    speed_mod_value = self.speed_mod.get_value(beat_pos) if self.speed_mod.enabled else 0.0
                    effective_speed = self.speed * (1.0 + speed_mod_value * 8.0)
                    frames_to_advance = delta_time * self.fps * effective_speed * seq_speed_mult
                    if self.reverse: self.playback_position -= frames_to_advance
                    else: self.playback_position += frames_to_advance
                    if self.loop: self.playback_position %= self.frame_count
                    else: self.playback_position = min(max(0, self.playback_position), self.frame_count - 1)
                    target_idx = int(self.playback_position)
                
                target_idx = max(0, min(self.frame_count - 1, int(target_idx)))
                
                if target_idx != self.current_frame_idx:
                    if target_idx in self.resized_cache:
                        self.current_resized = self.resized_cache[target_idx]
                    else:
                        cur_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                        if abs(cur_pos - target_idx) > 1:
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
                        ret, raw = self.cap.read()
                        if not ret:
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            ret, raw = self.cap.read()
                            if not ret: return None
                            target_idx = 0
                        self.current_resized = self._get_resized(target_idx, raw)
                    self.current_frame_idx = target_idx
                frame = self.current_resized

            if frame is None: return None

            if self.glitch_rate > 0.01 and random.random() < (self.glitch_rate * 0.4):
                frame = frame.copy()
                shift_amt = random.randint(-50, 50)
                axis = random.choice([0, 1])
                if random.randint(0, 3) == 0:
                    frame = np.roll(frame, shift_amt, axis=axis)
                else:
                    ch = random.randint(0, 2)
                    frame[:, :, ch] = np.roll(frame[:, :, ch], shift_amt, axis=axis)
            
            if self.rgb_mod.enabled and self.rgb_mod.depth > 0:
                val = self.rgb_mod.get_value(beat_pos) * 100.0 
                if abs(val) > 1:
                    frame = frame.copy()  # ALWAYS copy, don't condition on glitch_rate
                    shift = int(val)
                    frame[:, :, 0] = np.roll(frame[:, :, 0], shift, axis=1) 
                    frame[:, :, 2] = np.roll(frame[:, :, 2], -shift, axis=1)

        if self.pixel_mod.enabled and self.pixel_mod.depth > 0:
            pval = (self.pixel_mod.get_value(beat_pos) + 1.0) / 2.0 * 0.95 
            if pval > 0.05:
                frame = frame.copy()
                h, w = frame.shape[:2]
                factor = 1.0 - pval 
                small = cv2.resize(frame, (int(w*factor), int(h*factor)), interpolation=cv2.INTER_NEAREST)
                frame = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

        if self.zoom_mod.enabled and self.zoom_mod.depth > 0:
            zval = (self.zoom_mod.get_value(beat_pos)) * 0.5 
            scale = 1.0 + max(0.0, zval) 
            if scale > 1.01:
                frame = frame.copy()
                h, w = frame.shape[:2]
                M = cv2.getRotationMatrix2D((w/2, h/2), 0, scale)
                frame = cv2.warpAffine(frame, M, (w, h))

        if self.blur_mod.enabled and self.blur_mod.depth > 0:
            bval = (self.blur_mod.get_value(beat_pos) + 1.0) / 2.0 
            if bval > 0.05:
                k = int(bval * 30) * 2 + 1 
                frame = cv2.GaussianBlur(frame, (k, k), 0)

        if self.chroma_mod.enabled and self.chroma_mod.depth > 0:
            cval = self.chroma_mod.get_value(beat_pos) * 20.0  # Max offset of 20 pixels
            if abs(cval) > 0.5:
                frame = frame.copy()
                h, w = frame.shape[:2]
                offset = int(cval)
                
                # Create chromatic aberration by shifting color channels
                # Red channel: shift right+down
                M_r = np.float32([[1, 0, offset], [0, 1, offset]])
                frame[:, :, 2] = cv2.warpAffine(frame[:, :, 2], M_r, (w, h), borderMode=cv2.BORDER_WRAP)
                
                # Blue channel: shift left+up  
                M_b = np.float32([[1, 0, -offset], [0, 1, -offset]])
                frame[:, :, 0] = cv2.warpAffine(frame[:, :, 0], M_b, (w, h), borderMode=cv2.BORDER_WRAP)

        frame = self._apply_mirror(frame, beat_pos)

        # Apply mosh modulator to mosh_amount
        effective_mosh = self.mosh_amount
        if self.mosh_mod.enabled:
            mod_val = (self.mosh_mod.get_value(beat_pos) + 1.0) / 2.0  # Map -1..1 to 0..1
            effective_mosh = self.mosh_amount * mod_val

        if effective_mosh > 0.01:
            frame = frame.copy()
            if self.mosh_buffer is None or self.mosh_buffer.shape != frame.shape:
                self.mosh_buffer = frame.copy()
            else:
                alpha = 1.0 - effective_mosh
                self.mosh_buffer = cv2.addWeighted(frame, alpha, self.mosh_buffer, effective_mosh, 0)
            frame = self.mosh_buffer.copy()

        # Echo effect - creates motion trails by layering previous frames
        effective_echo = self.echo_amount
        if self.echo_mod.enabled:
            mod_val = (self.echo_mod.get_value(beat_pos) + 1.0) / 2.0
            effective_echo = self.echo_amount * mod_val

        if effective_echo > 0.01:
            frame = frame.copy()
            if self.echo_buffer is None or self.echo_buffer.shape != frame.shape:
                self.echo_buffer = frame.copy()
            else:
                # Extreme persistence - buffer barely decays
                # At max effective_echo, only 0.5% of new frame bleeds in
                decay = 0.05 - (effective_echo * 0.045)  # Range: 0.05 down to 0.005
                self.echo_buffer = cv2.addWeighted(frame, decay, self.echo_buffer, 1.0 - decay, 0)
            # Blend with massive echo presence
            current_weight = 0.6 - (effective_echo * 0.4)  # Range: 0.6 down to 0.2
            echo_weight = 0.6 + (effective_echo * 0.6)     # Range: 0.6 up to 1.2 (oversaturates)
            frame = cv2.addWeighted(frame, current_weight, self.echo_buffer, echo_weight, 0)

        # Slicer effect - creates scanline displacement glitches
        effective_slicer = self.slicer_amount
        if self.slicer_mod.enabled:
            mod_val = (self.slicer_mod.get_value(beat_pos) + 1.0) / 2.0
            effective_slicer = self.slicer_amount * mod_val

        if effective_slicer > 0.01:
            frame = frame.copy()
            h, w = frame.shape[:2]
            if self.slicer_buffer is None or self.slicer_buffer.shape != frame.shape:
                self.slicer_buffer = frame.copy()
            else:
                # Slice frame into horizontal strips
                num_slices = 16
                slice_height = h // num_slices
                for i in range(num_slices):
                    y_start = i * slice_height
                    y_end = min((i + 1) * slice_height, h)
                    # Randomly choose to use current or buffered slice
                    if random.random() < effective_slicer:
                        frame[y_start:y_end, :] = self.slicer_buffer[y_start:y_end, :]
                self.slicer_buffer = frame.copy()

        # Kaleidoscope effect - creates symmetrical mirror patterns
        effective_kaleidoscope = self.kaleidoscope_amount
        if self.kaleidoscope_mod.enabled:
            mod_val = (self.kaleidoscope_mod.get_value(beat_pos) + 1.0) / 2.0
            effective_kaleidoscope = self.kaleidoscope_amount * mod_val

        if effective_kaleidoscope > 0.01:
            frame = frame.copy()
            h, w = frame.shape[:2]
            
            # Simple mirror implementation for kaleidoscope
            quad = frame[:h//2, :w//2].copy()
            temp = np.zeros_like(frame)
            temp[:h//2, :w//2] = quad
            temp[:h//2, w//2:] = cv2.flip(quad, 1)
            temp[h//2:, :] = cv2.flip(temp[:h//2, :], 0)
            
            frame = cv2.addWeighted(frame, 1.0 - effective_kaleidoscope, temp, effective_kaleidoscope, 0)

        # Vignette effect - darkens edges
        effective_vignette = self.vignette_amount
        if self.vignette_mod.enabled:
            mod_val = (self.vignette_mod.get_value(beat_pos) + 1.0) / 2.0
            effective_vignette = self.vignette_amount * mod_val

        if effective_vignette > 0.01:
            frame = frame.copy()
            h, w = frame.shape[:2]
            
            # Create radial gradient mask
            y_indices, x_indices = np.ogrid[:h, :w]
            center_y, center_x = h / 2, w / 2
            
            # Calculate distance from center, normalized
            max_dist = np.sqrt(center_x**2 + center_y**2)
            distances = np.sqrt((x_indices - center_x)**2 + (y_indices - center_y)**2)
            distances = distances / max_dist
            
            # Create vignette mask (smooth falloff)
            # Allow vignette to extend all the way to center
            radius = 1.0 - effective_vignette
            if radius < 0.01:
                radius = 0.01  # Prevent radius from becoming zero for safe division below
            vignette_mask = np.clip(1.0 - (distances - radius) / (1.0 - radius), 0, 1)
            vignette_mask = vignette_mask ** 2
            
            # Apply transparency control to determine vignette strength
            # Note: Higher transparency value = stronger/more opaque vignette effect
            # At transparency=1.0: full vignette effect (most opaque)
            # At transparency=0.0: minimal vignette effect (most transparent)
            vignette_mask = 1.0 - (1.0 - vignette_mask) * self.vignette_transparency
            
            # Apply vignette
            for i in range(3):
                frame[:, :, i] = frame[:, :, i] * vignette_mask

        # Color Shift effect - HSV hue rotation
        effective_color_shift = self.color_shift_amount
        if self.color_shift_mod.enabled:
            mod_val = self.color_shift_mod.get_value(beat_pos)
            effective_color_shift = mod_val

        if abs(effective_color_shift) > 0.01:
            frame = frame.copy()
            # Convert to uint8 for HSV conversion
            frame_uint8 = (frame * 255.0).astype(np.uint8)
            # Convert to HSV
            hsv = cv2.cvtColor(frame_uint8, cv2.COLOR_BGR2HSV).astype(np.float32)
            
            # Shift hue (H channel is 0-179 in OpenCV)
            hue_shift = effective_color_shift * 90
            hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
            
            # Convert back to BGR
            frame_uint8 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            # Convert back to float32
            frame = frame_uint8.astype(np.float32) / 255.0

        # Spin effect - rotation from center
        effective_spin = self.spin_amount
        if self.spin_mod.enabled:
            mod_val = self.spin_mod.get_value(beat_pos)
            effective_spin = self.spin_amount * mod_val

        if abs(effective_spin) > 0.01:
            frame = frame.copy()
            h, w = frame.shape[:2]
            # Rotation angle based on spin amount (e.g., -180 to 180 degrees)
            angle = effective_spin * 180.0
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            frame = cv2.warpAffine(frame, M, (w, h))

        b = self.brightness + self.brightness_mod.get_value(beat_pos) * 0.5
        c = self.contrast + self.contrast_mod.get_value(beat_pos) * 0.5
        s = self.saturation + self.saturation_mod.get_value(beat_pos) * 0.5
        o = self.opacity + self.opacity_mod.get_value(beat_pos) * 0.5
        
        # Gate sequencer with timebase support
        seq_step = self._get_gate_step(beat_pos)
        gate_on = self.seq_gate[seq_step]
        
        # Trigger snare sound on gate transitions
        if self.gate_snare_enabled and gate_on and seq_step != self.last_gate_step and SNARE_SOUND:
            try:
                SNARE_SOUND.play()
            except:
                pass
        self.last_gate_step = seq_step
        
        if not gate_on:
            if self.gate_envelope_enabled:
                # Apply decay envelope when gate turns off
                step_duration_sec = self._get_step_timing(bpm)
                
                # Position within current step (0.0 to 1.0)
                step_pos = ((beat_pos % self.gate_timebase) * (16.0 / self.gate_timebase)) % 1.0
                
                # Calculate decay (gate is OFF, so apply release/decay)
                decay_time_sec = self.gate_decay if self.gate_decay > self.MIN_ENVELOPE_TIME else self.DEFAULT_ENVELOPE_TIME
                decay_progress = min(1.0, (step_pos * step_duration_sec) / decay_time_sec)
                o = o * (1.0 - decay_progress)
            else:
                o = 0.0
        elif self.gate_envelope_enabled:
            # Apply attack envelope when gate turns on
            step_duration_sec = self._get_step_timing(bpm)
            
            # Position within current step (0.0 to 1.0)
            step_pos = ((beat_pos % self.gate_timebase) * (16.0 / self.gate_timebase)) % 1.0
            
            # Calculate attack
            attack_time_sec = self.gate_attack if self.gate_attack > self.MIN_ENVELOPE_TIME else self.DEFAULT_ENVELOPE_TIME
            attack_progress = min(1.0, (step_pos * step_duration_sec) / attack_time_sec)
            o = o * attack_progress
            
        b = max(-1.0, min(1.0, b))
        c = max(0.1, min(3.0, c))
        s = max(0.0, min(2.0, s))
        o = max(0.0, min(1.0, o))
        
        needs_effects = (b != 0.0 or c != 1.0 or s != 1.0)
        if needs_effects:
            if not frame.flags['WRITEABLE']: frame = frame.copy() 
            frame = (frame + b - 0.5) * c + 0.5
            if s != 1.0:
                gray = frame[:,:,0]*0.114 + frame[:,:,1]*0.587 + frame[:,:,2]*0.299
                frame = gray[:,:,np.newaxis] + s * (frame - gray[:,:,np.newaxis])
        
        if self.strobe_enabled:
            phase = (beat_pos % self.strobe_rate) / self.strobe_rate
            if phase < 0.5:
                if self.strobe_color == 'white': frame = np.ones_like(frame)
                else: frame = np.zeros_like(frame)
                    
        return frame, o

class SequencerWidget(tk.Canvas):
    def __init__(self, parent, channel, attr_name, label, mode="toggle"):
        super().__init__(parent, height=25, bg="#333", highlightthickness=0)
        self.channel = channel
        self.attr_name = attr_name
        self.mode = mode 
        self.steps = getattr(channel, attr_name)
        self.rects = []
        self.bind("<Button-1>", self.on_click)
        if self.mode == "toggle": self.colors = ["#444", "#0f0"]
        elif self.mode == "multi_speed": self.colors = ["#444", "#ff0", "#00f", "#f00", "#000"]
        elif self.mode == "multi_jump": self.colors = ["#444", "#ff0", "#f00"] 
        w = 15
        h = 20
        self.width = w * 16
        self.configure(width=self.width + 50)
        self.create_text(25, 12, text=label, fill="white", anchor="e", font=("Arial", 8))
        start_x = 30
        for i in range(16):
            x = start_x + i * w
            val = self.steps[i]
            if val >= len(self.colors): val = 0
            r = self.create_rectangle(x, 2, x + w - 1, 2 + h, fill=self.colors[val], outline="black")
            self.rects.append(r)
            if i % 4 == 0: self.create_line(x, 0, x, 25, fill="#777")

    def on_click(self, event):
        x = event.x - 30
        if x < 0: return
        idx = x // 15
        if 0 <= idx < 16:
            current = self.steps[idx]
            nxt = (current + 1) % len(self.colors)
            self.steps[idx] = nxt
            self.update_ui()
    def update_ui(self):
        for i, r in enumerate(self.rects):
            val = self.steps[i]
            if val >= len(self.colors): val = 0
            self.itemconfigure(r, fill=self.colors[val])

class TimelineWidget(tk.Canvas):
    """Interactive timeline widget with professional waveform visualization and draggable loop points."""
    
    def __init__(self, parent, mixer):
        super().__init__(parent, height=120, bg="#1a1a1a", highlightthickness=1, highlightbackground="#555")
        self.mixer = mixer
        self.waveform_samples = None  # Raw normalized samples
        self.sample_rate = 44100
        self.duration_sec = 0
        
        # Dragging state
        self.dragging = None  # 'start', 'end', or None
        self.drag_offset = 0
        
        # Bind mouse events
        self.bind("<Button-1>", self.on_mouse_down)
        self.bind("<B1-Motion>", self.on_mouse_drag)
        self.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.bind("<Configure>", self.on_resize)
        
        self.draw_empty()
    
    def draw_empty(self):
        """Draw an empty timeline when no audio is loaded."""
        self.delete("all")
        w = self.winfo_width() or 800
        h = self.winfo_height() or 120
        self.create_text(w // 2, h // 2, text="No Audio Loaded", fill="#666", font=("Arial", 12))
    
    def load_audio_waveform(self, audio_path):
        """Extract and cache waveform data from audio file using min/max approach."""
        try:
            # Load audio as a pygame Sound object to extract samples
            # Note: pygame.sndarray works best with WAV files; other formats may need conversion
            sound = pygame.mixer.Sound(audio_path)
            self.duration_sec = sound.get_length()
            
            # Extract sample data using pygame.sndarray
            samples = pygame.sndarray.array(sound)
            
            # Convert to mono if stereo (average channels)
            if samples.ndim > 1 and samples.shape[1] > 1:
                samples = np.mean(samples, axis=1)
            
            # Normalize to -1 to 1 range
            samples = samples.astype(np.float32)
            max_abs_sample = np.max(np.abs(samples))
            if max_abs_sample > 0:
                samples = samples / max_abs_sample
            
            # Store raw samples for min/max rendering
            self.waveform_samples = samples
            
            self.redraw()
            
        except Exception as e:
            print(f"Failed to load waveform: {e}")
            # Clear waveform but don't crash
            self.waveform_samples = None
            self.duration_sec = 0
            self.draw_empty()
    
    def redraw(self):
        """Redraw the entire timeline."""
        self.delete("all")
        
        if self.waveform_samples is None or len(self.waveform_samples) == 0:
            self.draw_empty()
            return
        
        w = self.winfo_width()
        h = self.winfo_height()
        
        if w <= 1 or h <= 1:
            return
        
        # Draw waveform
        self.draw_waveform(w, h)
        
        # Draw grid
        self.draw_grid(w, h)
        
        # Draw loop handles
        self.draw_loop_handles(w, h)
        
        # Draw playhead (always, even when paused)
        self.draw_playhead(w, h)
    
    def draw_waveform(self, w, h):
        """Draw the audio waveform using min/max rendering for professional look."""
        if self.waveform_samples is None:
            return
        
        # Dark background
        self.create_rectangle(0, 0, w, h, fill="#0a0a0a", outline="")
        
        # Center zero-line
        mid_y = h // 2
        self.create_line(0, mid_y, w, mid_y, fill="#333333", width=1)
        
        # Scale for amplitude
        amp_scale = (h // 2) * 0.85
        
        # Min/Max rendering: for each pixel column, find min/max samples
        samples = self.waveform_samples
        # Use float for step to maintain precision across the full width
        step = len(samples) / float(max(1, w))
        
        for x in range(w):
            idx_start = int(x * step)
            idx_end = int((x + 1) * step)
            
            # Ensure we don't go out of bounds
            idx_end = min(idx_end, len(samples))
            
            if idx_start >= len(samples):
                break
                
            # Get chunk of samples for this pixel column
            chunk = samples[idx_start:idx_end]
            
            if len(chunk) > 0:
                min_v = np.min(chunk)
                max_v = np.max(chunk)
                
                # Calculate y coordinates (inverted because canvas y increases downward)
                y_min = mid_y - (max_v * amp_scale)
                y_max = mid_y - (min_v * amp_scale)
                
                # Draw vertical line from min to max (creates solid waveform)
                self.create_line(x, y_min, x, y_max, fill="#00CCFF", width=1)
    
    def draw_grid(self, w, h):
        """Draw grid lines representing beats (with emphasis on bars)."""
        if self.duration_sec <= 0:
            return
        
        bpm = self.mixer.bpm
        beats_per_bar = self.mixer.beats_per_bar
        
        # Calculate beat duration in seconds
        beat_duration_sec = 60.0 / bpm
        
        # Draw vertical lines for each beat
        beat_num = 0
        while True:
            beat_time_sec = beat_num * beat_duration_sec
            if beat_time_sec > self.duration_sec:
                break
            
            x = (beat_time_sec / self.duration_sec) * w
            
            # Emphasize bar lines (every beats_per_bar beats)
            is_bar = (beat_num % beats_per_bar) == 0
            color = "#666" if is_bar else "#333"
            width = 2 if is_bar else 1
            self.create_line(x, 0, x, h, fill=color, width=width, tags="grid")
            
            # Draw bar number labels for bar lines
            if is_bar:
                bar_num = beat_num // beats_per_bar
                self.create_text(x + 2, 10, text=f"B{bar_num}", fill="#888", anchor="nw", font=("Arial", 8), tags="grid")
            
            beat_num += 1
    
    def draw_loop_handles(self, w, h):
        """Draw the loop start and end handles as thick bars with flag markers."""
        if self.duration_sec <= 0:
            return
        
        bpm = self.mixer.bpm
        beat_duration_sec = 60.0 / bpm
        
        # Calculate positions (now using beats directly)
        start_time_sec = self.mixer.global_loop_start * beat_duration_sec
        end_time_sec = self.mixer.global_loop_end * beat_duration_sec
        
        start_x = (start_time_sec / self.duration_sec) * w
        end_x = (end_time_sec / self.duration_sec) * w
        
        # Draw loop region (shaded area) - use stipple pattern for transparency effect
        self.create_rectangle(start_x, 0, end_x, h, fill="#00ff00", stipple="gray12", outline="", tags="loop_region")
        
        # Draw START handle - thick green bar with flag
        handle_width = 8
        flag_width = 40
        flag_height = 20
        
        # Start handle bar
        self.create_rectangle(start_x - handle_width//2, 0, start_x + handle_width//2, h, 
                             fill="#00ff00", outline="", tags="loop_start")
        
        # Start flag at top
        self.create_rectangle(start_x + handle_width//2, 5, start_x + handle_width//2 + flag_width, 5 + flag_height,
                             fill="#00ff00", outline="", tags="loop_start")
        self.create_text(start_x + handle_width//2 + flag_width//2, 5 + flag_height//2, 
                        text="S", fill="#000000", font=("Arial", 12, "bold"), tags="loop_start")
        
        # Draw END handle - thick red bar with flag
        # End handle bar
        self.create_rectangle(end_x - handle_width//2, 0, end_x + handle_width//2, h,
                             fill="#ff0000", outline="", tags="loop_end")
        
        # End flag at top
        self.create_rectangle(end_x - handle_width//2 - flag_width, 5, end_x - handle_width//2, 5 + flag_height,
                             fill="#ff0000", outline="", tags="loop_end")
        self.create_text(end_x - handle_width//2 - flag_width//2, 5 + flag_height//2,
                        text="E", fill="#000000", font=("Arial", 12, "bold"), tags="loop_end")
    
    def draw_playhead(self, w, h):
        """Draw the current playback position - ALWAYS visible."""
        if self.duration_sec <= 0:
            return
        
        # Calculate current playback position from mixer.beat_position
        if hasattr(self.mixer, 'beat_position'):
            beat_pos = self.mixer.beat_position
            
            bpm = self.mixer.bpm
            beat_duration_sec = 60.0 / bpm
            
            time_sec = beat_pos * beat_duration_sec
            x = (time_sec / self.duration_sec) * w
            
            # Draw playhead line - bright yellow, always visible
            self.create_line(x, 0, x, h, fill="#ffff00", width=3, tags="playhead")
    
    def update_playhead(self):
        """Update only the playhead position (called frequently during playback)."""
        if self.duration_sec <= 0:
            return
        
        w = self.winfo_width()
        h = self.winfo_height()
        
        if w <= 1 or h <= 1:
            return
        
        # Calculate current playback position
        if hasattr(self.mixer, 'beat_position'):
            beat_pos = self.mixer.beat_position
            bar_pos = beat_pos / self.mixer.beats_per_bar
            
            bpm = self.mixer.bpm
            beats_per_bar = self.mixer.beats_per_bar
            bar_duration_sec = (60.0 / bpm) * beats_per_bar
            
            time_sec = bar_pos * bar_duration_sec
            x = (time_sec / self.duration_sec) * w
            
            # Update playhead position
            self.delete("playhead")
            self.create_line(x, 0, x, h, fill="#ffff00", width=3, tags="playhead")
    
    def _is_click_on_start_handle(self, event_x, event_y, start_x):
        """Check if click is on the start handle (bar or flag)."""
        handle_width = 8
        flag_width = 40
        click_tolerance = max(handle_width, 15)
        
        # Check if clicked on the handle bar or flag
        on_bar = abs(event_x - start_x) < click_tolerance
        on_flag = (event_x >= start_x + handle_width//2 and 
                   event_x <= start_x + handle_width//2 + flag_width and 
                   event_y <= 25)
        return on_bar or on_flag
    
    def _is_click_on_end_handle(self, event_x, event_y, end_x):
        """Check if click is on the end handle (bar or flag)."""
        handle_width = 8
        flag_width = 40
        click_tolerance = max(handle_width, 15)
        
        # Check if clicked on the handle bar or flag
        on_bar = abs(event_x - end_x) < click_tolerance
        on_flag = (event_x >= end_x - handle_width//2 - flag_width and 
                   event_x <= end_x - handle_width//2 and 
                   event_y <= 25)
        return on_bar or on_flag
    
    def on_mouse_down(self, event):
        """Handle mouse button press with improved wider handle detection."""
        if self.duration_sec <= 0:
            return
        
        w = self.winfo_width()
        bpm = self.mixer.bpm
        beat_duration_sec = 60.0 / bpm
        
        # Calculate positions (now using beats)
        start_time_sec = self.mixer.global_loop_start * beat_duration_sec
        end_time_sec = self.mixer.global_loop_end * beat_duration_sec
        
        start_x = (start_time_sec / self.duration_sec) * w
        end_x = (end_time_sec / self.duration_sec) * w
        
        # Check which handle was clicked using helper methods
        if self._is_click_on_start_handle(event.x, event.y, start_x):
            self.dragging = 'start'
            self.drag_offset = event.x - start_x
        elif self._is_click_on_end_handle(event.x, event.y, end_x):
            self.dragging = 'end'
            self.drag_offset = event.x - end_x
        else:
            # Click on timeline to seek - set playhead position immediately
            # (no dragging involved for seek operation)
            time_sec = (event.x / w) * self.duration_sec
            beat_pos = time_sec / beat_duration_sec
            # Snap to nearest beat
            snapped_beat = max(0, round(beat_pos))
            
            # Update the mixer's beat position
            if hasattr(self.mixer, 'beat_position'):
                self.mixer.beat_position = snapped_beat
                # Redraw playhead immediately
                self.delete("playhead")
                canvas_width = self.winfo_width()
                canvas_height = self.winfo_height()
                self.draw_playhead(canvas_width, canvas_height)
    
    def on_mouse_drag(self, event):
        """Handle mouse drag for loop handle dragging only."""
        if self.dragging is None or self.duration_sec <= 0:
            return
        
        w = self.winfo_width()
        bpm = self.mixer.bpm
        beat_duration_sec = 60.0 / bpm
        
        # Calculate time position from mouse x
        adjusted_x = event.x - self.drag_offset
        time_sec = (adjusted_x / w) * self.duration_sec
        
        # Convert to beat position
        beat_pos = time_sec / beat_duration_sec
        
        # Snap to nearest beat
        snapped_beat = round(beat_pos)
        snapped_beat = max(0, snapped_beat)
        
        # Update the appropriate loop point
        if self.dragging == 'start':
            # Ensure start < end
            if snapped_beat < self.mixer.global_loop_end:
                self.mixer.global_loop_start = snapped_beat
                self.mixer.gloop_start.set(snapped_beat)
        elif self.dragging == 'end':
            # Ensure end > start
            if snapped_beat > self.mixer.global_loop_start:
                self.mixer.global_loop_end = snapped_beat
                self.mixer.gloop_end.set(snapped_beat)
        
        # Redraw loop handles
        canvas_width = self.winfo_width()
        canvas_height = self.winfo_height()
        self.delete("loop_region")
        self.delete("loop_start")
        self.delete("loop_end")
        self.draw_loop_handles(canvas_width, canvas_height)
    
    def on_mouse_up(self, event):
        """Handle mouse button release."""
        self.dragging = None
        self.drag_offset = 0
    
    def on_resize(self, event):
        """Handle canvas resize."""
        self.redraw()

class VideoProcessor(threading.Thread):
    def __init__(self, mixer):
        super().__init__(daemon=True)
        self.mixer = mixer
        self.running = False
        self.frame_queue = queue.Queue(maxsize=2)
        self.start_time = 0
        self.lock = threading.Lock()
    
    def run(self):
        try:
            import ctypes
            k = ctypes.windll.kernel32
            k.SetThreadPriority(k.GetCurrentThread(), 15)
            k.SetPriorityClass(k.GetCurrentProcess(), 0x80)
        except: pass
        last = time.perf_counter()
        
        while self.running:
            if not self.mixer.playing or not self.mixer.sync_ready:
                time.sleep(0.001)
                last = time.perf_counter()
                continue
            now = time.perf_counter()
            dt = now - last
            last = now
            with self.lock: st = self.start_time
            if st <= 0:
                time.sleep(0.001)
                continue
            
            # Use offset from GUI (convert ms to seconds)
            offset_sec = self.mixer.latency_ms.get() / 1000.0
            
            # --- TIGHT SYNC CORE ---
            # Visuals run on (now - start_time - offset)
            # This allows shifting the visual clock backwards or forwards relative to audio
            effective_time = (now - st) - offset_sec
            if effective_time < 0: effective_time = 0
            
            total_beats = effective_time * self.mixer.bpm / 60.0
            
            if self.mixer.global_loop_enabled:
                # Loop markers are now in beats
                loop_len_beats = self.mixer.global_loop_end - self.mixer.global_loop_start
                if loop_len_beats <= 0: loop_len_beats = 16
                start_beats = self.mixer.global_loop_start
                
                # Simple time-based loop detection
                # Calculate elapsed time in milliseconds
                elapsed_ms = (now - st) * 1000.0
                
                # Calculate loop boundaries in ms (using beats directly)
                beat_duration_ms = (60.0 / self.mixer.bpm) * 1000.0
                loop_start_ms = self.mixer.global_loop_start * beat_duration_ms
                
                # Drift correction: sync to audio clock if active
                if self.mixer.audio_track.is_active():
                    audio_pos = self.mixer.audio_track.get_time_ms()
                    expected_pos = loop_start_ms + elapsed_ms
                    drift = audio_pos - expected_pos
                    if abs(drift) > 10:
                        # Apply 10% correction factor to smooth out jitter
                        with self.lock:
                            self.start_time -= (drift / 1000.0) * 0.1
                        # Recalculate elapsed_ms after adjustment
                        elapsed_ms = (now - self.start_time) * 1000.0
                
                loop_end_ms = self.mixer.global_loop_end * beat_duration_ms
                loop_duration_ms = loop_end_ms - loop_start_ms
                
                # If we've passed the loop end, jump back
                if elapsed_ms >= loop_duration_ms:
                    # Restart audio at loop start
                    if self.mixer.audio_track.enabled:
                        self.mixer.audio_track.play(int(loop_start_ms))
                    # Reset the clock so visuals also jump back
                    with self.lock: 
                        self.start_time = now
                    # Set beat position to loop start
                    total_beats = start_beats
                    # Reset metronome to one beat before loop start so check_tick() detects the downbeat
                    # (check_tick uses: if int(beat_pos) > int(current_beat) to trigger ticks)
                    self.mixer.metronome.current_beat = start_beats - 1.0
                else:
                    # Within loop range - calculate beat position from elapsed time
                    rel_beats = (elapsed_ms / 1000.0) * (self.mixer.bpm / 60.0)
                    total_beats = start_beats + rel_beats

            bp = total_beats
            
            # Metronome needs raw time, but we feed it the corrected time
            self.mixer.metronome.check_tick(bp)
            
            fa, oa, fb, ob = None, 1.0, None, 1.0
            if self.mixer.channel_a.cap:
                r = self.mixer.channel_a.get_frame(bp, dt, self.mixer.bpm, self.mixer.beats_per_bar)
                if r: fa, oa = r
            if self.mixer.channel_b.cap:
                r = self.mixer.channel_b.get_frame(bp, dt, self.mixer.bpm, self.mixer.beats_per_bar)
                if r: fb, ob = r
            if fa is not None or fb is not None:
                blended = self.mixer.blend_frames(fa, oa, fb, ob, bp)
                try:
                    while not self.frame_queue.empty(): self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait((blended, bp))
                except: pass
            time.sleep(0.001)
    
    def start_proc(self, st):
        with self.lock: self.start_time = st
        if not self.is_alive():
            self.running = True
            self.start()
    def stop_proc(self):
        self.running = False
        with self.lock: self.start_time = 0
        while not self.frame_queue.empty():
            try: self.frame_queue.get_nowait()
            except: break
    def get_frame(self):
        try: return self.frame_queue.get_nowait()
        except: return None

class VideoMixer:
    BLEND_MODES = ["normal", "add", "multiply", "screen", "overlay", "difference",
                   "exclusion", "hard_light", "soft_light", "color_dodge", "color_burn",
                   "darken", "lighten", "linear_light", "pin_light", "vivid_light"]
    
    def __init__(self, root):
        self.root = root
        self.root.title("BPM Video Mixer v15 Fixed")
        self.root.geometry("1400x950")
        try:
            import ctypes
            ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x80)
        except: pass
        self.preview_width = 640
        self.preview_height = 360
        self.channel_a = VideoChannel(self.preview_width, self.preview_height)
        self.channel_b = VideoChannel(self.preview_width, self.preview_height)
        
        self.audio_track = AudioChannel()
        self.global_loop_enabled = True
        self.global_loop_start = 0  # Now in beats instead of bars
        self.global_loop_end = 16   # 4 bars * 4 beats_per_bar = 16 beats
        self.loop_trigger_flag = False
        
        self.bpm = 120.0
        self.beats_per_bar = 4
        self.mix = 0.5
        self.mix_mod = Modulator()
        self.blend_mode = "normal"
        self.playing = False
        self.start_time = 0
        self.beat_position = 0
        self.last_update_time = time.perf_counter()
        self.metronome = HighPrecisionMetronome()
        self.sync_ready = False
        self.processor = VideoProcessor(self)
        self.blend_buffer = np.zeros((self.preview_height, self.preview_width, 3), dtype=np.float32)
        self.output_buffer = np.zeros((self.preview_height, self.preview_width, 3), dtype=np.uint8)
        self.status = tk.StringVar(value="Ready")
        self.setup_ui()
        self.update_loop()
        
    def trigger_audio_loop(self):
        # Manually restart audio from the beginning
        # Note: This is not used by the automatic loop logic
        if self.audio_track.enabled:
            self.audio_track.play(0)

    def reset_mod(self, c):
        c['en'].set(False)
        c['wv'].set("sine")
        c['rt'].set("1")
        c['dp'].set(1.0)
        if 'pos' in c: c['pos'].set(False)
        if 'neg' in c: c['neg'].set(False)
        c['inv'].set(False)
        if 'spd_fwd' in c: c['spd_fwd'].set(False)
        if 'spd_rev' in c: c['spd_rev'].set(False)

    def reset_all(self):
        for ch, c in [(self.channel_a, self.ch_a), (self.channel_b, self.ch_b)]:
            ch.reset_controls()
            c['br_v'].set(0)
            c['co_v'].set(1)
            c['sa_v'].set(1)
            c['op_v'].set(1)
            c['sp_v'].set("1")
            c['loop'].set(True)
            c['rev'].set(False)
            c['glitch'].set(0.0)
            c['bl_en'].set(False)
            c['bl_len_var'].set("1 bar")
            c['bl_start'].set(0)
            c['strobe_en'].set(False)
            c['strobe_rt'].set("1/8")
            c['post_rt'].set("Off")
            c['mirror_mode'].set("Off")
            c['mosh'].set(0.0)
            c['echo'].set(0.0)
            c['slicer'].set(0.0)
            c['kaleidoscope'].set(0.0)
            c['vignette'].set(0.0)
            c['vignette_transparency'].set(0.5)
            c['color_shift'].set(0.0)
            c['spin'].set(0.0)
            if ch.frame_count > 0:
                c['bl_lbl'].config(text=f"0/{ch.frame_count}")
            for k in ['br_m', 'co_m', 'sa_m', 'op_m']:
                self.reset_mod(c[f'{k}_m'])
            for k in ['loop_start_mod', 'rgb_mod', 'blur_mod', 'zoom_mod', 'pixel_mod', 'chroma_mod', 'mirror_center_mod', 'speed_mod', 'kaleidoscope_mod', 'vignette_mod', 'color_shift_mod', 'spin_mod', 'mosh_mod', 'echo_mod', 'slicer_mod']:
                self.reset_mod(c[k])
            c['seq_gate_w'].update_ui()
            c['seq_stutter_w'].update_ui()
            c['seq_speed_w'].update_ui()
            c['seq_jump_w'].update_ui()
        self.mix_var.set(0.5)
        self.mix = 0.5
        self.mix_mod.reset()
        self.reset_mod(self.mix_mod_c)
        self.blend_var.set("normal")
        self.blend_mode = "normal"
        self.mbright.set(0)
        self.mcontr.set(1)
        self.metro_var.set(False)
        self.metronome.enabled = False
        self.mvol.set(0.5)
        self.bpb_var.set(4)
        self.beats_per_bar = 4
        self.gloop_en.set(True)
        self.global_loop_enabled = True
        self.gloop_start.set(0)
        self.global_loop_start = 0
        self.gloop_end.set(16)  # Reset to 16 beats (4 bars)
        self.global_loop_end = 16
        self.audio_en.set(True)
        self.audio_track.enabled = True
        self.latency_ms.set(0.0)
        self.status.set("Reset")

    def reset_parameters(self):
        """Reset all parameters but keep videos loaded"""
        for ch, c in [(self.channel_a, self.ch_a), (self.channel_b, self.ch_b)]:
            # Store the video path and loaded state before reset
            video_path = ch.video_path
            cap_opened = ch.cap is not None and ch.cap.isOpened() if ch.cap else False
            
            # Reset all controls (this resets parameters to defaults)
            ch.reset_controls()
            
            # Restore video if it was loaded
            if cap_opened and video_path:
                ch.load_video(video_path)
            
            # Reset Bonus tab UI controls
            c['kaleidoscope'].set(0.0)
            c['vignette'].set(0.0)
            c['vignette_transparency'].set(0.5)
            c['color_shift'].set(0.0)
            c['spin'].set(0.0)
            # Reset all mod controls on bonus tab
            for mod_key in ['rgb_mod', 'blur_mod', 'zoom_mod', 'pixel_mod', 'chroma_mod', 
                            'kaleidoscope_mod', 'vignette_mod', 'color_shift_mod', 'spin_mod']:
                if mod_key in c:
                    self.reset_mod(c[mod_key])
            
            # Reset Seq tab UI controls
            # The underlying data is reset by reset_controls(), but we need to update the UI
            c['seq_gate_w'].steps[:] = [1] * 16
            c['seq_gate_w'].update_ui()
            c['seq_stutter_w'].steps[:] = [0] * 16
            c['seq_stutter_w'].update_ui()
            c['seq_speed_w'].steps[:] = [0] * 16
            c['seq_speed_w'].update_ui()
            c['seq_jump_w'].steps[:] = [0] * 16
            c['seq_jump_w'].update_ui()
            
            # Update all UI controls to reflect reset values
            self.update_ch_ui(ch, c)
        
        # Reset mixer parameters
        self.mix_var.set(0.5)
        self.mix = 0.5
        self.mix_mod.reset()
        self.reset_mod(self.mix_mod_c)
        self.blend_var.set("normal")
        self.blend_mode = "normal"
        self.mbright.set(0)
        self.mcontr.set(1)
        
        # Reset metronome controls
        self.metro_var.set(False)
        self.metronome.enabled = False
        self.mvol.set(0.5)
        
        self.status.set("Parameters Reset (Videos Kept)")

    def setup_ui(self):
        main = ttk.Frame(self.root, padding="5")
        main.pack(fill=tk.BOTH, expand=True)
        top = ttk.Frame(main)
        top.pack(fill=tk.X, pady=(0, 5))
        pf = ttk.LabelFrame(top, text="Preview", padding="5")
        pf.pack(side=tk.LEFT, padx=(0, 10))
        self.preview_canvas = tk.Canvas(pf, width=self.preview_width, height=self.preview_height, bg="black")
        self.preview_canvas.pack()
        tf = ttk.LabelFrame(top, text="Transport", padding="5")
        tf.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Row 1
        row1 = ttk.Frame(tf)
        row1.pack(fill=tk.X, pady=2)
        ttk.Label(row1, text="BPM:", font=("Arial", 11, "bold")).pack(side=tk.LEFT)
        self.bpm_var = tk.DoubleVar(value=120.0)
        bpm_spin = ttk.Spinbox(row1, from_=20, to=300, textvariable=self.bpm_var, width=7, command=self.on_bpm)
        bpm_spin.pack(side=tk.LEFT, padx=3)
        bpm_spin.bind("<Return>", lambda e: self.on_bpm())
        self.tap_times = []
        ttk.Button(row1, text="Tap", command=self.tap_tempo, width=4).pack(side=tk.LEFT, padx=3)
        ttk.Label(row1, text="BPB:").pack(side=tk.LEFT, padx=(5, 0))
        self.bpb_var = tk.IntVar(value=4)
        ttk.Spinbox(row1, from_=1, to=16, textvariable=self.bpb_var, width=3, command=self.on_bpb).pack(side=tk.LEFT)
        self.beat_var = tk.StringVar(value="Beat: 0.0")
        ttk.Label(row1, textvariable=self.beat_var, font=("Consolas", 10)).pack(side=tk.LEFT, padx=10)
        self.beat_flash = tk.Canvas(row1, width=25, height=25, bg="gray", highlightthickness=1)
        self.beat_flash.pack(side=tk.LEFT, padx=3)
        self.sync_var = tk.StringVar(value="")
        ttk.Label(row1, textvariable=self.sync_var, foreground="green").pack(side=tk.LEFT, padx=5)
        self.fps_var = tk.StringVar(value="")
        ttk.Label(row1, textvariable=self.fps_var, font=("Consolas", 9)).pack(side=tk.RIGHT)
        
        # Row 2
        row2 = ttk.Frame(tf)
        row2.pack(fill=tk.X, pady=2)
        ttk.Label(row2, text="Aspect:").pack(side=tk.LEFT, padx=(0, 2))
        self.aspect_var = tk.StringVar(value="16:9")
        aspect_combo = ttk.Combobox(row2, textvariable=self.aspect_var, values=["16:9", "4:3", "1:1", "3:4", "9:16"], state="readonly", width=6)
        aspect_combo.pack(side=tk.LEFT, padx=3)
        aspect_combo.bind("<<ComboboxSelected>>", lambda e: self.on_aspect_ratio_change())
        ttk.Label(row2, text=" | ").pack(side=tk.LEFT)
        self.metro_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row2, text="Metro", variable=self.metro_var, command=lambda: setattr(self.metronome, 'enabled', self.metro_var.get())).pack(side=tk.LEFT)
        ttk.Label(row2, text="Vol:").pack(side=tk.LEFT, padx=(5, 2))
        self.mvol = tk.DoubleVar(value=0.5)
        ttk.Scale(row2, from_=0, to=1, variable=self.mvol, command=lambda v: self.metronome.update_volume(float(v)), length=50).pack(side=tk.LEFT)
        
        ttk.Label(row2, text=" | Loop Beats:").pack(side=tk.LEFT, padx=5)
        self.gloop_en = tk.BooleanVar(value=True)
        ttk.Checkbutton(row2, variable=self.gloop_en, command=lambda: setattr(self, 'global_loop_enabled', self.gloop_en.get())).pack(side=tk.LEFT)
        self.gloop_start = tk.IntVar(value=0)
        ttk.Spinbox(row2, from_=0, to=400, textvariable=self.gloop_start, width=3, command=lambda: setattr(self, 'global_loop_start', self.gloop_start.get())).pack(side=tk.LEFT)
        ttk.Label(row2, text="to").pack(side=tk.LEFT)
        self.gloop_end = tk.IntVar(value=16)
        ttk.Spinbox(row2, from_=1, to=400, textvariable=self.gloop_end, width=3, command=lambda: setattr(self, 'global_loop_end', self.gloop_end.get())).pack(side=tk.LEFT)
        
        ttk.Label(row2, text=" | Audio:").pack(side=tk.LEFT, padx=5)
        self.audio_en = tk.BooleanVar(value=True)
        ttk.Checkbutton(row2, variable=self.audio_en, command=lambda: setattr(self.audio_track, 'enabled', self.audio_en.get())).pack(side=tk.LEFT)
        ttk.Button(row2, text="Load", command=self.load_audio, width=5).pack(side=tk.LEFT)
        self.audio_status = tk.StringVar(value="None")
        ttk.Label(row2, textvariable=self.audio_status, width=10).pack(side=tk.LEFT, padx=2)
        
        # Row 3
        row3 = ttk.Frame(tf)
        row3.pack(fill=tk.X, pady=5)
        self.play_btn = tk.Button(row3, text="Play", command=self.toggle_play, font=("Arial", 11), width=8)
        self.play_btn.pack(side=tk.LEFT, padx=2)
        tk.Button(row3, text="Stop", command=self.stop, font=("Arial", 11), width=8).pack(side=tk.LEFT, padx=2)
        tk.Button(row3, text="Rew", command=self.rewind, font=("Arial", 11), width=6).pack(side=tk.LEFT, padx=2)
        tk.Button(row3, text="Reset", command=self.reset_all, font=("Arial", 11), width=6).pack(side=tk.LEFT, padx=2)
        tk.Button(row3, text="Param Reset", command=self.reset_parameters, font=("Arial", 11), width=10).pack(side=tk.LEFT, padx=2)
        
        ttk.Label(row3, text="Offset (ms):").pack(side=tk.LEFT, padx=(10, 5))
        self.latency_ms = tk.DoubleVar(value=0.0)
        ttk.Scale(row3, from_=-200, to=200, variable=self.latency_ms, length=100).pack(side=tk.LEFT)
        
        # Row 4
        row4 = ttk.Frame(tf)
        row4.pack(fill=tk.X, pady=2)
        ttk.Label(row4, text="A", font=("Arial", 11, "bold")).pack(side=tk.LEFT)
        self.mix_var = tk.DoubleVar(value=0.5)
        ttk.Scale(row4, from_=0, to=1, variable=self.mix_var, command=lambda v: setattr(self, 'mix', float(v)), length=150).pack(side=tk.LEFT, padx=3)
        ttk.Label(row4, text="B", font=("Arial", 11, "bold")).pack(side=tk.LEFT)
        self.mix_mod_c = self.setup_mod(row4, self.mix_mod, "Mix")
        
        # Row 5
        row5 = ttk.Frame(tf)
        row5.pack(fill=tk.X, pady=2)
        ttk.Label(row5, text="Blend:").pack(side=tk.LEFT)
        self.blend_var = tk.StringVar(value="normal")
        blend_combo = ttk.Combobox(row5, textvariable=self.blend_var, values=self.BLEND_MODES, state="readonly", width=12)
        blend_combo.pack(side=tk.LEFT, padx=3)
        blend_combo.bind("<<ComboboxSelected>>", lambda e: setattr(self, 'blend_mode', self.blend_var.get()))
        ttk.Label(row5, text="Bright:").pack(side=tk.LEFT, padx=(10, 2))
        self.mbright = tk.DoubleVar(value=0.0)
        ttk.Scale(row5, from_=-1, to=1, variable=self.mbright, length=60).pack(side=tk.LEFT)
        ttk.Label(row5, text="Contr:").pack(side=tk.LEFT, padx=(5, 2))
        self.mcontr = tk.DoubleVar(value=1.0)
        ttk.Scale(row5, from_=0, to=2, variable=self.mcontr, length=60).pack(side=tk.LEFT)
        
        # Row 6
        row6 = ttk.Frame(tf)
        row6.pack(fill=tk.X, pady=2)
        ttk.Button(row6, text="Load Proj", command=self.load_project).pack(side=tk.LEFT, padx=2)
        ttk.Button(row6, text="Save Proj", command=self.save_project).pack(side=tk.LEFT, padx=2)
        ttk.Button(row6, text="Load Preset", command=self.load_preset).pack(side=tk.LEFT, padx=2)
        ttk.Button(row6, text="Save Preset", command=self.save_preset).pack(side=tk.LEFT, padx=2)
        tk.Button(row6, text="Export", command=self.export_video, font=("Arial", 10)).pack(side=tk.LEFT, padx=10)
        
        # Timeline Widget
        timeline_frame = ttk.LabelFrame(main, text="Timeline", padding="5")
        timeline_frame.pack(fill=tk.X, pady=(5, 0))
        self.timeline_widget = TimelineWidget(timeline_frame, self)
        self.timeline_widget.pack(fill=tk.BOTH, expand=True)
        
        # Status Bar
        self.status_bar = ttk.Label(main, textvariable=self.status, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(5,0))
        
        chf = ttk.Frame(main)
        chf.pack(fill=tk.BOTH, expand=True)
        self.ch_a = self.setup_channel(chf, self.channel_a, "Channel A")
        self.ch_b = self.setup_channel(chf, self.channel_b, "Channel B")
        
    def on_bpm(self):
        try:
            self.bpm = float(self.bpm_var.get())
            self.metronome.set_bpm(self.bpm)
            # Update timeline widget when BPM changes
            if hasattr(self, 'timeline_widget'):
                self.timeline_widget.redraw()
        except:
            pass
        
    def tap_tempo(self):
        now = time.perf_counter()
        self.tap_times = [t for t in self.tap_times if now - t < 3] + [now]
        if len(self.tap_times) >= 2:
            avg = sum(self.tap_times[i] - self.tap_times[i-1] for i in range(1, len(self.tap_times))) / (len(self.tap_times) - 1)
            self.bpm = max(20, min(300, 60 / avg))
            self.bpm_var.set(round(self.bpm, 1))
            self.metronome.set_bpm(self.bpm)
        
    def on_bpb(self):
        try:
            self.beats_per_bar = int(self.bpb_var.get())
            self.metronome.set_beats_per_bar(self.beats_per_bar)
        except:
            pass
    
    def on_aspect_ratio_change(self):
        """Handle aspect ratio change from the dropdown."""
        aspect = self.aspect_var.get()
        
        # Define aspect ratio mappings
        aspect_ratios = {
            "16:9": (16, 9),
            "4:3": (4, 3),
            "1:1": (1, 1),
            "3:4": (3, 4),
            "9:16": (9, 16)
        }
        
        if aspect not in aspect_ratios:
            return
        
        w_ratio, h_ratio = aspect_ratios[aspect]
        
        # Keep width at 640 and adjust height to maintain aspect ratio
        base_width = 640
        new_height = int(base_width * h_ratio / w_ratio)
        
        # Only update if dimensions actually changed
        if self.preview_width == base_width and self.preview_height == new_height:
            return
        
        # Update preview dimensions
        self.preview_width = base_width
        self.preview_height = new_height
        
        # Update canvas size
        self.preview_canvas.config(width=self.preview_width, height=self.preview_height)
        
        # Update video channels target size
        self.channel_a.set_target_size(self.preview_width, self.preview_height)
        self.channel_b.set_target_size(self.preview_width, self.preview_height)
        
        # Reinitialize buffers with new dimensions
        self.blend_buffer = np.zeros((self.preview_height, self.preview_width, 3), dtype=np.float32)
        self.output_buffer = np.zeros((self.preview_height, self.preview_width, 3), dtype=np.uint8)
        
        self.status.set(f"Aspect ratio changed to {aspect}")
    
    def load_audio(self):
        p = filedialog.askopenfilename(filetypes=[("Audio", "*.wav *.mp3 *.ogg")])
        if not p:
            return
            
        # Load waveform into timeline widget FIRST to avoid file lock issues
        if hasattr(self, 'timeline_widget'):
            self.timeline_widget.load_audio_waveform(p)
            
        if self.audio_track.load(p):
            self.audio_status.set(os.path.basename(p)[:10])
            self.status.set(f"Audio Loaded: {os.path.basename(p)}")
            
            # Auto-calculate loop end based on audio duration
            duration_ms = self.audio_track.engine.duration_ms
            if duration_ms > 0 and self.bpm > 0 and self.beats_per_bar > 0:
                # Calculate duration of one beat in seconds
                beat_duration_sec = 60.0 / self.bpm
                # Convert file duration to seconds
                duration_sec = duration_ms / 1000.0
                # Round to nearest whole beat, minimum 1 beat
                num_beats = max(1, round(duration_sec / beat_duration_sec))
                # Update global_loop_end and UI widget (now in beats)
                self.global_loop_end = int(num_beats)
                self.gloop_end.set(int(num_beats))
                # Trigger timeline redraw to show correct loop region
                if hasattr(self, 'timeline_widget'):
                    self.timeline_widget.redraw()

    def setup_channel(self, parent, ch, title):
        c = {}
        f = ttk.LabelFrame(parent, text=title, padding="2")
        f.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=3)
        nb = ttk.Notebook(f)
        nb.pack(fill=tk.BOTH, expand=True)
        tab_main = ttk.Frame(nb, padding=5)
        tab_loop = ttk.Frame(nb, padding=5)
        tab_fx = ttk.Frame(nb, padding=5)
        tab_seq = ttk.Frame(nb, padding=5)
        tab_bonus = ttk.Frame(nb, padding=5)
        nb.add(tab_main, text="Main")
        nb.add(tab_loop, text="Loop/Time")
        nb.add(tab_fx, text="FX")
        nb.add(tab_seq, text="Seq")
        nb.add(tab_bonus, text="Bonus")
        
        row1 = ttk.Frame(tab_main)
        row1.pack(fill=tk.X, pady=2)
        # Fix: Helper method restored
        ttk.Button(row1, text="Load", command=lambda: self.load_video_helper(ch, c)).pack(side=tk.LEFT, padx=2)
        ttk.Button(row1, text="Rew", command=ch.reset_position).pack(side=tk.LEFT, padx=2)
        c['file'] = tk.StringVar(value="No video")
        ttk.Label(row1, textvariable=c['file']).pack(side=tk.LEFT, padx=5)
        c['info'] = tk.StringVar(value="")
        ttk.Label(tab_main, textvariable=c['info'], font=("Consolas", 8)).pack(anchor=tk.W)
        for k, l, d, mn, mx, a in [('br', 'Bright', 0, -1, 1, 'brightness'), ('co', 'Contr', 1, 0, 2, 'contrast'), 
                                    ('sa', 'Satur', 1, 0, 2, 'saturation'), ('op', 'Opac', 1, 0, 1, 'opacity')]:
            row = ttk.Frame(tab_main)
            row.pack(fill=tk.X, pady=1)
            ttk.Label(row, text=f"{l}:", width=6).pack(side=tk.LEFT)
            c[f'{k}_v'] = tk.DoubleVar(value=d)
            ttk.Scale(row, from_=mn, to=mx, variable=c[f'{k}_v'], length=70, command=lambda v, a=a, ch=ch: setattr(ch, a, float(v))).pack(side=tk.LEFT)
            c[f'{k}_m'] = self.setup_mod(row, getattr(ch, f'{a}_mod'), "M")
            
        srow = ttk.Frame(tab_loop)
        srow.pack(fill=tk.X, pady=5)
        ttk.Label(srow, text="Speed:", width=6).pack(side=tk.LEFT)
        c['sp_v'] = tk.StringVar(value="1")
        speed_combo = ttk.Combobox(srow, textvariable=c['sp_v'], values=VideoChannel.SPEED_LABELS, state="readonly", width=5)
        speed_combo.pack(side=tk.LEFT, padx=2)
        speed_combo.bind("<<ComboboxSelected>>", lambda e, ch=ch: self.on_speed_change(e, ch))
        c['loop'] = tk.BooleanVar(value=True)
        ttk.Checkbutton(srow, text="Loop", variable=c['loop'], command=lambda: setattr(ch, 'loop', c['loop'].get())).pack(side=tk.LEFT, padx=3)
        c['rev'] = tk.BooleanVar(value=False)
        ttk.Checkbutton(srow, text="Rev", variable=c['rev'], command=lambda: setattr(ch, 'reverse', c['rev'].get())).pack(side=tk.LEFT, padx=3)
        srow2 = ttk.Frame(tab_loop)
        srow2.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(srow2, text="Speed Mod:", font=("Arial", 8)).pack(side=tk.LEFT)
        c['speed_mod'] = self.setup_mod_simple(srow2, ch.speed_mod, "On")
        mf = c['speed_mod']['_frame']
        c['spd_fwd'] = tk.BooleanVar(value=False)
        c['spd_rev'] = tk.BooleanVar(value=False)
        ttk.Checkbutton(mf, text="Fwd", variable=c['spd_fwd'], command=lambda: self._set_speed_dir(ch.speed_mod, c, 'fwd')).pack(side=tk.LEFT)
        ttk.Checkbutton(mf, text="Rev", variable=c['spd_rev'], command=lambda: self._set_speed_dir(ch.speed_mod, c, 'rev')).pack(side=tk.LEFT)
        lf = ttk.LabelFrame(tab_loop, text="Beat Loop", padding="3")
        lf.pack(fill=tk.X, pady=5)
        lrow = ttk.Frame(lf)
        lrow.pack(fill=tk.X)
        c['bl_en'] = tk.BooleanVar(value=False)
        ttk.Checkbutton(lrow, text="On", variable=c['bl_en'], command=lambda: setattr(ch, 'beat_loop_enabled', c['bl_en'].get())).pack(side=tk.LEFT)
        ttk.Label(lrow, text="Len:").pack(side=tk.LEFT, padx=5)
        c['bl_len_var'] = tk.StringVar(value="1 bar")
        bl_combo = ttk.Combobox(lrow, textvariable=c['bl_len_var'], values=VideoChannel.LOOP_LENGTH_LABELS, state="readonly", width=8)
        bl_combo.pack(side=tk.LEFT)
        bl_combo.bind("<<ComboboxSelected>>", lambda e, ch=ch: self.on_loop_len_change(e, ch))
        lrow2 = ttk.Frame(lf)
        lrow2.pack(fill=tk.X, pady=2)
        ttk.Label(lrow2, text="Start:").pack(side=tk.LEFT)
        c['bl_start'] = tk.IntVar(value=0)
        c['bl_slider'] = ttk.Scale(lrow2, from_=0, to=100, variable=c['bl_start'], length=120, command=lambda v, ch=ch, c=c: self.on_ls(ch, c))
        c['bl_slider'].pack(side=tk.LEFT)
        c['bl_lbl'] = ttk.Label(lrow2, text="0/0", width=8)
        c['bl_lbl'].pack(side=tk.LEFT)
        lrow3 = ttk.Frame(lf)
        lrow3.pack(fill=tk.X, pady=(2, 0))
        ttk.Label(lrow3, text="Mod:", font=("Arial", 8)).pack(side=tk.LEFT)
        c['loop_start_mod'] = self.setup_mod_simple(lrow3, ch.loop_start_mod, "On")
        lrow4 = ttk.Frame(lf)
        lrow4.pack(fill=tk.X, pady=(2, 0))
        ttk.Label(lrow4, text="Fine Tune:", font=("Arial", 8)).pack(side=tk.LEFT)
        c['fine_tune'] = tk.DoubleVar(value=ch.loop_start_mod.fine_tune)
        ttk.Scale(lrow4, from_=0, to=100, variable=c['fine_tune'], length=120, command=lambda v, ch=ch: setattr(ch.loop_start_mod, 'fine_tune', float(v))).pack(side=tk.LEFT)
        c['fine_tune_lbl'] = ttk.Label(lrow4, text=f"{ch.loop_start_mod.fine_tune:.2f}", width=6)
        c['fine_tune_lbl'].pack(side=tk.LEFT)
        c['fine_tune'].trace_add('write', lambda *args: c['fine_tune_lbl'].config(text=f"{c['fine_tune'].get():.2f}"))

        fr_str = ttk.Frame(tab_fx)
        fr_str.pack(fill=tk.X, pady=2)
        c['strobe_en'] = tk.BooleanVar(value=False)
        ttk.Checkbutton(fr_str, text="Strobe", variable=c['strobe_en'], command=lambda: setattr(ch, 'strobe_enabled', c['strobe_en'].get())).pack(side=tk.LEFT)
        c['strobe_rt'] = tk.StringVar(value="1/8")
        sc = ttk.Combobox(fr_str, textvariable=c['strobe_rt'], values=list(VideoChannel.STROBE_RATES.keys()), state="readonly", width=4)
        sc.pack(side=tk.LEFT, padx=5)
        sc.bind("<<ComboboxSelected>>", lambda e, ch=ch, v=c['strobe_rt']: setattr(ch, 'strobe_rate', VideoChannel.STROBE_RATES.get(v.get(), 0.125)))
        c['strobe_col'] = tk.StringVar(value="white")
        scc = ttk.Combobox(fr_str, textvariable=c['strobe_col'], values=["white", "black"], state="readonly", width=5)
        scc.pack(side=tk.LEFT)
        scc.bind("<<ComboboxSelected>>", lambda e, ch=ch, v=c['strobe_col']: setattr(ch, 'strobe_color', v.get()))
        fr_mir = ttk.Frame(tab_fx)
        fr_mir.pack(fill=tk.X, pady=2)
        ttk.Label(fr_mir, text="Mirror:").pack(side=tk.LEFT)
        c['mirror_mode'] = tk.StringVar(value="Off")
        mc = ttk.Combobox(fr_mir, textvariable=c['mirror_mode'], values=VideoChannel.MIRROR_MODES, state="readonly", width=9)
        mc.pack(side=tk.LEFT, padx=5)
        mc.bind("<<ComboboxSelected>>", lambda e, ch=ch, v=c['mirror_mode']: setattr(ch, 'mirror_mode', v.get()))
        mir_row = ttk.Frame(tab_fx)
        mir_row.pack(fill=tk.X, pady=(0, 2))
        ttk.Label(mir_row, text="Mod:", font=("Arial", 8)).pack(side=tk.LEFT)
        c['mirror_center_mod'] = self.setup_mod_simple(mir_row, ch.mirror_center_mod, "On")
        fr_mosh = ttk.Frame(tab_fx)
        fr_mosh.pack(fill=tk.X, pady=2)
        ttk.Label(fr_mosh, text="Mosh:").pack(side=tk.LEFT)
        c['mosh'] = tk.DoubleVar(value=0.0)
        ttk.Scale(fr_mosh, from_=0.0, to=1.0, variable=c['mosh'], length=80, command=lambda v, ch=ch: setattr(ch, 'mosh_amount', float(v))).pack(side=tk.LEFT)
        c['mosh_mod'] = self.setup_mod_simple(fr_mosh, ch.mosh_mod, "LFO")
        fr_echo = ttk.Frame(tab_fx)
        fr_echo.pack(fill=tk.X, pady=2)
        ttk.Label(fr_echo, text="Echo:").pack(side=tk.LEFT)
        c['echo'] = tk.DoubleVar(value=0.0)
        ttk.Scale(fr_echo, from_=0.0, to=1.0, variable=c['echo'], length=80, command=lambda v, ch=ch: setattr(ch, 'echo_amount', float(v))).pack(side=tk.LEFT)
        c['echo_mod'] = self.setup_mod_simple(fr_echo, ch.echo_mod, "LFO")
        fr_slicer = ttk.Frame(tab_fx)
        fr_slicer.pack(fill=tk.X, pady=2)
        ttk.Label(fr_slicer, text="Slicer:").pack(side=tk.LEFT)
        c['slicer'] = tk.DoubleVar(value=0.0)
        ttk.Scale(fr_slicer, from_=0.0, to=1.0, variable=c['slicer'], length=80, command=lambda v, ch=ch: setattr(ch, 'slicer_amount', float(v))).pack(side=tk.LEFT)
        c['slicer_mod'] = self.setup_mod_simple(fr_slicer, ch.slicer_mod, "LFO")
        fr_post = ttk.Frame(tab_fx)
        fr_post.pack(fill=tk.X, pady=2)
        ttk.Label(fr_post, text="Postrz:").pack(side=tk.LEFT)
        c['post_rt'] = tk.StringVar(value="Off")
        pc = ttk.Combobox(fr_post, textvariable=c['post_rt'], values=list(VideoChannel.POSTERIZE_RATES.keys()), state="readonly", width=5)
        pc.pack(side=tk.LEFT, padx=5)
        pc.bind("<<ComboboxSelected>>", lambda e, ch=ch, v=c['post_rt']: setattr(ch, 'posterize_rate', VideoChannel.POSTERIZE_RATES.get(v.get(), 0.0)))
        fr_gli = ttk.Frame(tab_fx)
        fr_gli.pack(fill=tk.X, pady=2)
        ttk.Label(fr_gli, text="Glitch:").pack(side=tk.LEFT)
        c['glitch'] = tk.DoubleVar(value=0.0)
        ttk.Scale(fr_gli, from_=0.0, to=1.0, variable=c['glitch'], length=80, command=lambda v, ch=ch: setattr(ch, 'glitch_rate', float(v))).pack(side=tk.LEFT)
        fr_spin = ttk.Frame(tab_fx)
        fr_spin.pack(fill=tk.X, pady=2)
        ttk.Label(fr_spin, text="Spin:").pack(side=tk.LEFT)
        c['spin'] = tk.DoubleVar(value=0.0)
        ttk.Scale(fr_spin, from_=0.0, to=1.0, variable=c['spin'], length=80, command=lambda v, ch=ch: setattr(ch, 'spin_amount', float(v))).pack(side=tk.LEFT)
        c['spin_mod'] = self.setup_mod_simple(fr_spin, ch.spin_mod, "LFO")

        # Bonus Tab Effects
        # Move LFO effects to Bonus tab for better 4:3 aspect ratio compatibility
        fr_rgb = ttk.Frame(tab_bonus)
        fr_rgb.pack(fill=tk.X, pady=2)
        ttk.Label(fr_rgb, text="RGB LFO:").pack(side=tk.LEFT)
        c['rgb_mod'] = self.setup_mod(fr_rgb, ch.rgb_mod, "On")
        fr_blur = ttk.Frame(tab_bonus)
        fr_blur.pack(fill=tk.X, pady=2)
        ttk.Label(fr_blur, text="Blur LFO:").pack(side=tk.LEFT)
        c['blur_mod'] = self.setup_mod(fr_blur, ch.blur_mod, "On")
        fr_zoom = ttk.Frame(tab_bonus)
        fr_zoom.pack(fill=tk.X, pady=2)
        ttk.Label(fr_zoom, text="Zoom LFO:").pack(side=tk.LEFT)
        c['zoom_mod'] = self.setup_mod(fr_zoom, ch.zoom_mod, "On")
        fr_pix = ttk.Frame(tab_bonus)
        fr_pix.pack(fill=tk.X, pady=2)
        ttk.Label(fr_pix, text="Pixel LFO:").pack(side=tk.LEFT)
        c['pixel_mod'] = self.setup_mod(fr_pix, ch.pixel_mod, "On")
        fr_chroma = ttk.Frame(tab_bonus)
        fr_chroma.pack(fill=tk.X, pady=2)
        ttk.Label(fr_chroma, text="Chroma LFO:").pack(side=tk.LEFT)
        c['chroma_mod'] = self.setup_mod_simple(fr_chroma, ch.chroma_mod, "On")

        fr_kaleido = ttk.Frame(tab_bonus)
        fr_kaleido.pack(fill=tk.X, pady=2)
        ttk.Label(fr_kaleido, text="Kaleid:").pack(side=tk.LEFT)
        c['kaleidoscope'] = tk.DoubleVar(value=0.0)
        ttk.Scale(fr_kaleido, from_=0.0, to=1.0, variable=c['kaleidoscope'], length=80, 
                  command=lambda v, ch=ch: setattr(ch, 'kaleidoscope_amount', float(v))).pack(side=tk.LEFT)
        c['kaleidoscope_mod'] = self.setup_mod_simple(fr_kaleido, ch.kaleidoscope_mod, "LFO")

        fr_vignette = ttk.Frame(tab_bonus)
        fr_vignette.pack(fill=tk.X, pady=2)
        ttk.Label(fr_vignette, text="Vignette:").pack(side=tk.LEFT)
        c['vignette'] = tk.DoubleVar(value=0.0)
        ttk.Scale(fr_vignette, from_=0.0, to=1.0, variable=c['vignette'], length=80,
                  command=lambda v, ch=ch: setattr(ch, 'vignette_amount', float(v))).pack(side=tk.LEFT)
        c['vignette_mod'] = self.setup_mod_simple(fr_vignette, ch.vignette_mod, "LFO")

        fr_vignette_trans = ttk.Frame(tab_bonus)
        fr_vignette_trans.pack(fill=tk.X, pady=2)
        ttk.Label(fr_vignette_trans, text="Vig Trans:").pack(side=tk.LEFT)
        c['vignette_transparency'] = tk.DoubleVar(value=0.5)
        ttk.Scale(fr_vignette_trans, from_=0.0, to=1.0, variable=c['vignette_transparency'], length=80,
                  command=lambda v, ch=ch: setattr(ch, 'vignette_transparency', float(v))).pack(side=tk.LEFT)
        c['vignette_trans_lbl'] = ttk.Label(fr_vignette_trans, text="0.50", width=5)
        c['vignette_trans_lbl'].pack(side=tk.LEFT)
        c['vignette_transparency'].trace_add('write', lambda *args: c['vignette_trans_lbl'].config(text=f"{c['vignette_transparency'].get():.2f}"))

        fr_colorshift = ttk.Frame(tab_bonus)
        fr_colorshift.pack(fill=tk.X, pady=2)
        ttk.Label(fr_colorshift, text="ColorShift:").pack(side=tk.LEFT)
        c['color_shift'] = tk.DoubleVar(value=0.0)
        ttk.Scale(fr_colorshift, from_=-1.0, to=1.0, variable=c['color_shift'], length=80,
                  command=lambda v, ch=ch: setattr(ch, 'color_shift_amount', float(v))).pack(side=tk.LEFT)
        c['color_shift_mod'] = self.setup_mod_simple(fr_colorshift, ch.color_shift_mod, "LFO")

        c['seq_gate_w'] = SequencerWidget(tab_seq, ch, 'seq_gate', "Gate", "toggle")
        c['seq_gate_w'].pack(pady=5)
        
        # Gate sequencer controls
        gate_controls_frame = ttk.Frame(tab_seq)
        gate_controls_frame.pack(fill=tk.X, pady=2)
        
        # Snare checkbox
        c['gate_snare'] = tk.BooleanVar(value=False)
        ttk.Checkbutton(gate_controls_frame, text="Snare", variable=c['gate_snare'], 
                       command=lambda: setattr(ch, 'gate_snare_enabled', c['gate_snare'].get())).pack(side=tk.LEFT, padx=5)
        
        # Envelope checkbox
        c['gate_env'] = tk.BooleanVar(value=False)
        ttk.Checkbutton(gate_controls_frame, text="Env", variable=c['gate_env'], 
                       command=lambda: setattr(ch, 'gate_envelope_enabled', c['gate_env'].get())).pack(side=tk.LEFT, padx=5)
        
        # Timebase dropdown
        ttk.Label(gate_controls_frame, text="Timebase:").pack(side=tk.LEFT, padx=5)
        c['gate_timebase'] = tk.StringVar(value="4")
        timebase_options = ["1/32", "1/16", "1/8", "1/4", "1/2", "1", "2", "4"]
        timebase_values = {"1/32": 0.03125, "1/16": 0.0625, "1/8": 0.125, "1/4": 0.25, 
                          "1/2": 0.5, "1": 1.0, "2": 2.0, "4": 4.0}
        timebase_combo = ttk.Combobox(gate_controls_frame, textvariable=c['gate_timebase'], 
                                     values=timebase_options, state="readonly", width=5)
        timebase_combo.pack(side=tk.LEFT, padx=2)
        timebase_combo.bind("<<ComboboxSelected>>", 
                           lambda e: setattr(ch, 'gate_timebase', timebase_values.get(c['gate_timebase'].get(), 4.0)))
        
        # Attack and Decay sliders
        gate_env_frame = ttk.Frame(tab_seq)
        gate_env_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(gate_env_frame, text="Attack:").pack(side=tk.LEFT, padx=5)
        c['gate_attack'] = tk.DoubleVar(value=0.0)
        ttk.Scale(gate_env_frame, from_=0.0, to=1.0, variable=c['gate_attack'], length=100,
                 command=lambda v: setattr(ch, 'gate_attack', float(v))).pack(side=tk.LEFT)
        c['gate_attack_lbl'] = ttk.Label(gate_env_frame, text="0.00", width=5)
        c['gate_attack_lbl'].pack(side=tk.LEFT)
        c['gate_attack'].trace_add('write', lambda *args: c['gate_attack_lbl'].config(text=f"{c['gate_attack'].get():.2f}"))
        
        ttk.Label(gate_env_frame, text="Decay:").pack(side=tk.LEFT, padx=5)
        c['gate_decay'] = tk.DoubleVar(value=0.0)
        ttk.Scale(gate_env_frame, from_=0.0, to=1.0, variable=c['gate_decay'], length=100,
                 command=lambda v: setattr(ch, 'gate_decay', float(v))).pack(side=tk.LEFT)
        c['gate_decay_lbl'] = ttk.Label(gate_env_frame, text="0.00", width=5)
        c['gate_decay_lbl'].pack(side=tk.LEFT)
        c['gate_decay'].trace_add('write', lambda *args: c['gate_decay_lbl'].config(text=f"{c['gate_decay'].get():.2f}"))
        
        c['seq_stutter_w'] = SequencerWidget(tab_seq, ch, 'seq_stutter', "Stutter", "toggle")
        c['seq_stutter_w'].pack(pady=5)
        c['seq_speed_w'] = SequencerWidget(tab_seq, ch, 'seq_speed', "Speed (G=1x,Y=2x,B=.5,R=Rev)", "multi_speed")
        c['seq_speed_w'].pack(pady=5)
        c['seq_jump_w'] = SequencerWidget(tab_seq, ch, 'seq_jump', "Jump (Y=-1bt, R=-1bar)", "multi_jump")
        c['seq_jump_w'].pack(pady=5)
        return c
    
    def setup_mod_simple(self, parent, mod, label):
        c = {}
        mf = ttk.Frame(parent)
        mf.pack(side=tk.LEFT, padx=2)
        c['_frame'] = mf
        c['en'] = tk.BooleanVar(value=False)
        ttk.Checkbutton(mf, text=label, variable=c['en'], command=lambda: setattr(mod, 'enabled', c['en'].get())).pack(side=tk.LEFT)
        c['wv'] = tk.StringVar(value="sine")
        wc = ttk.Combobox(mf, textvariable=c['wv'], values=Modulator.WAVE_TYPES, state="readonly", width=5)
        wc.pack(side=tk.LEFT)
        mod.wave_type = c['wv'].get()
        wc.bind("<<ComboboxSelected>>", lambda e: setattr(mod, 'wave_type', c['wv'].get()))
        c['rt'] = tk.StringVar(value="1")
        rc = ttk.Combobox(mf, textvariable=c['rt'], values=list(Modulator.RATE_OPTIONS.keys()), state="readonly", width=4)
        rc.pack(side=tk.LEFT)
        rc.bind("<<ComboboxSelected>>", lambda e: setattr(mod, 'rate', Modulator.RATE_OPTIONS.get(c['rt'].get(), 1.0)))
        c['dp'] = tk.DoubleVar(value=1.0)
        ttk.Scale(mf, from_=0, to=1, variable=c['dp'], length=80, command=lambda v: setattr(mod, 'depth', float(v))).pack(side=tk.LEFT)
        c['dp_lbl'] = ttk.Label(mf, text=f"{1.0:.2f}", width=5)
        c['dp_lbl'].pack(side=tk.LEFT)
        c['dp'].trace_add('write', lambda *args: c['dp_lbl'].config(text=f"{c['dp'].get():.2f}"))
        c['inv'] = tk.BooleanVar(value=False)
        ttk.Checkbutton(mf, text="inv", variable=c['inv'], command=lambda: setattr(mod, 'invert', c['inv'].get())).pack(side=tk.LEFT)
        return c
        
    def setup_mod(self, parent, mod, label):
        c = {}
        mf = ttk.Frame(parent)
        mf.pack(side=tk.LEFT, padx=2)
        c['en'] = tk.BooleanVar(value=False)
        ttk.Checkbutton(mf, text=label, variable=c['en'], command=lambda: setattr(mod, 'enabled', c['en'].get())).pack(side=tk.LEFT)
        c['wv'] = tk.StringVar(value="sine")
        wc = ttk.Combobox(mf, textvariable=c['wv'], values=Modulator.WAVE_TYPES, state="readonly", width=4)
        wc.pack(side=tk.LEFT)
        mod.wave_type = c['wv'].get()
        wc.bind("<<ComboboxSelected>>", lambda e: setattr(mod, 'wave_type', c['wv'].get()))
        c['rt'] = tk.StringVar(value="1")
        rc = ttk.Combobox(mf, textvariable=c['rt'], values=list(Modulator.RATE_OPTIONS.keys()), state="readonly", width=4)
        rc.pack(side=tk.LEFT)
        rc.bind("<<ComboboxSelected>>", lambda e: setattr(mod, 'rate', Modulator.RATE_OPTIONS.get(c['rt'].get(), 1.0)))
        c['dp'] = tk.DoubleVar(value=1.0)
        ttk.Scale(mf, from_=0, to=1, variable=c['dp'], length=80, command=lambda v: setattr(mod, 'depth', float(v))).pack(side=tk.LEFT)
        c['dp_lbl'] = ttk.Label(mf, text=f"{1.0:.2f}", width=5)
        c['dp_lbl'].pack(side=tk.LEFT)
        c['dp'].trace_add('write', lambda *args: c['dp_lbl'].config(text=f"{c['dp'].get():.2f}"))
        c['pos'] = tk.BooleanVar(value=False)
        c['neg'] = tk.BooleanVar(value=False)
        c['inv'] = tk.BooleanVar(value=False)
        ttk.Checkbutton(mf, text="+", variable=c['pos'], command=lambda: self._set_pos_neg(mod, c, 'pos')).pack(side=tk.LEFT)
        ttk.Checkbutton(mf, text="-", variable=c['neg'], command=lambda: self._set_pos_neg(mod, c, 'neg')).pack(side=tk.LEFT)
        ttk.Checkbutton(mf, text="inv", variable=c['inv'], command=lambda: setattr(mod, 'invert', c['inv'].get())).pack(side=tk.LEFT)
        return c
    
    def _set_pos_neg(self, mod, c, which):
        if which == 'pos':
            mod.pos_only = c['pos'].get()
            if mod.pos_only:
                mod.neg_only = False
                c['neg'].set(False)
        else:
            mod.neg_only = c['neg'].get()
            if mod.neg_only:
                mod.pos_only = False
                c['pos'].set(False)
    
    def _set_speed_dir(self, mod, c, which):
        if which == 'fwd':
            mod.fwd_only = c['spd_fwd'].get()
            if mod.fwd_only:
                c['spd_rev'].set(False)
                mod.rev_only = False
        elif which == 'rev':
            mod.rev_only = c['spd_rev'].get()
            if mod.rev_only:
                c['spd_fwd'].set(False)
                mod.fwd_only = False
    
    def on_ls(self, ch, c):
        v = int(float(c['bl_start'].get()))
        ch.loop_start_frame = v
        c['bl_lbl'].config(text=f"{v}/{ch.frame_count}")
    
    def on_speed_change(self, event, ch):
        val = event.widget.get()
        if val in VideoChannel.SPEED_OPTIONS:
            with ch.lock:
                ch.speed = float(VideoChannel.SPEED_OPTIONS[val])
        self.root.focus_set()
                
    def on_loop_len_change(self, event, ch):
        val = event.widget.get()
        if val in VideoChannel.LOOP_LENGTH_OPTIONS:
            with ch.lock:
                ch.loop_length_beats = float(VideoChannel.LOOP_LENGTH_OPTIONS[val])
        self.root.focus_set()
        
    def load_video_helper(self, ch, c):
        p = filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv *.webm")])
        if p and ch.load_video(p):
            c['file'].set(os.path.basename(p)[:30])
            c['info'].set(f"{ch.fps:.0f}fps {ch.width}x{ch.height} {ch.frame_count}f")
            c['bl_slider'].config(to=max(1, ch.frame_count - 1))
            c['bl_start'].set(0)
            c['bl_lbl'].config(text=f"0/{ch.frame_count}")
            self.status.set(f"Loaded: {os.path.basename(p)}")
    
    def update_ch_ui(self, ch, c):
        c['br_v'].set(ch.brightness)
        c['co_v'].set(ch.contrast)
        c['sa_v'].set(ch.saturation)
        c['op_v'].set(ch.opacity)
        for label, val in VideoChannel.SPEED_OPTIONS.items():
            if abs(val - ch.speed) < 0.01:
                c['sp_v'].set(label)
                break
        c['loop'].set(ch.loop)
        c['rev'].set(ch.reverse)
        c['glitch'].set(ch.glitch_rate)
        c['strobe_en'].set(ch.strobe_enabled)
        c['strobe_col'].set(ch.strobe_color)
        for label, val in VideoChannel.STROBE_RATES.items():
            if abs(val - ch.strobe_rate) < 0.001:
                c['strobe_rt'].set(label)
                break
        for label, val in VideoChannel.POSTERIZE_RATES.items():
            if abs(val - ch.posterize_rate) < 0.001:
                c['post_rt'].set(label)
                break
        c['mirror_mode'].set(ch.mirror_mode)
        c['mosh'].set(ch.mosh_amount)
        c['echo'].set(ch.echo_amount)
        c['slicer'].set(ch.slicer_amount)
        c['bl_en'].set(ch.beat_loop_enabled)
        for label, val in VideoChannel.LOOP_LENGTH_OPTIONS.items():
             if abs(val - ch.loop_length_beats) < 0.001:
                 c['bl_len_var'].set(label)
                 break
        c['bl_start'].set(ch.loop_start_frame)
        if ch.frame_count > 0:
            c['bl_slider'].config(to=max(1, ch.frame_count - 1))
            c['bl_lbl'].config(text=f"{ch.loop_start_frame}/{ch.frame_count}")
        # Update Bonus tab sliders
        c['kaleidoscope'].set(ch.kaleidoscope_amount)
        c['vignette'].set(ch.vignette_amount)
        c['vignette_transparency'].set(ch.vignette_transparency)
        c['color_shift'].set(ch.color_shift_amount)
        c['spin'].set(ch.spin_amount)
        for k, m in [('br', ch.brightness_mod), ('co', ch.contrast_mod), ('sa', ch.saturation_mod), ('op', ch.opacity_mod)]:
            mc = c[f'{k}_m']
            mc['en'].set(m.enabled)
            mc['wv'].set(m.wave_type)
            mc['rt'].set(Modulator.RATE_REVERSE.get(m.rate, "1"))
            mc['dp'].set(m.depth)
            mc['pos'].set(m.pos_only)
            mc['neg'].set(m.neg_only)
            mc['inv'].set(m.invert)
        for m, k in [(ch.loop_start_mod, 'loop_start_mod'), (ch.rgb_mod, 'rgb_mod'), 
                     (ch.blur_mod, 'blur_mod'), (ch.zoom_mod, 'zoom_mod'), (ch.pixel_mod, 'pixel_mod'),
                     (ch.chroma_mod, 'chroma_mod'), (ch.mosh_mod, 'mosh_mod'), 
                     (ch.echo_mod, 'echo_mod'), (ch.slicer_mod, 'slicer_mod'),
                     (ch.mirror_center_mod, 'mirror_center_mod'), (ch.speed_mod, 'speed_mod'),
                     (ch.kaleidoscope_mod, 'kaleidoscope_mod'), (ch.vignette_mod, 'vignette_mod'),
                     (ch.color_shift_mod, 'color_shift_mod'), (ch.spin_mod, 'spin_mod')]:
            mc = c[k]
            mc['en'].set(m.enabled)
            mc['wv'].set(m.wave_type)
            mc['rt'].set(Modulator.RATE_REVERSE.get(m.rate, "1"))
            mc['dp'].set(m.depth)
            mc['inv'].set(m.invert)
            # Update pos_only and neg_only if these controls exist (for rgb, blur, zoom, pixel)
            if 'pos' in mc:
                mc['pos'].set(m.pos_only)
            if 'neg' in mc:
                mc['neg'].set(m.neg_only)
        # Update fine tune control for loop start modulator
        if 'fine_tune' in c:
            c['fine_tune'].set(ch.loop_start_mod.fine_tune)
        if 'spd_fwd' in c:
            c['spd_fwd'].set(ch.speed_mod.fwd_only)
        if 'spd_rev' in c:
            c['spd_rev'].set(ch.speed_mod.rev_only)
        c['seq_gate_w'].update_ui()
        c['seq_stutter_w'].update_ui()
        c['seq_speed_w'].update_ui()
        c['seq_jump_w'].update_ui()
        # Update gate sequencer controls
        if 'gate_snare' in c:
            c['gate_snare'].set(ch.gate_snare_enabled)
        if 'gate_env' in c:
            c['gate_env'].set(ch.gate_envelope_enabled)
        if 'gate_attack' in c:
            c['gate_attack'].set(ch.gate_attack)
        if 'gate_decay' in c:
            c['gate_decay'].set(ch.gate_decay)
        if 'gate_timebase' in c:
            # Find matching timebase label
            timebase_values = {"1/32": 0.03125, "1/16": 0.0625, "1/8": 0.125, "1/4": 0.25, 
                              "1/2": 0.5, "1": 1.0, "2": 2.0, "4": 4.0}
            for label, val in timebase_values.items():
                if abs(val - ch.gate_timebase) < 0.001:
                    c['gate_timebase'].set(label)
                    break
        if ch.video_path:
            c['file'].set(os.path.basename(ch.video_path)[:30])
            c['info'].set(f"{ch.fps:.0f}fps {ch.width}x{ch.height} {ch.frame_count}f")
    
    def update_mod_ui(self, c, m):
        c['en'].set(m.enabled)
        c['wv'].set(m.wave_type)
        c['rt'].set(Modulator.RATE_REVERSE.get(m.rate, "1"))
        c['dp'].set(m.depth)
        if 'pos' in c:
            c['pos'].set(m.pos_only)
        if 'neg' in c:
            c['neg'].set(m.neg_only)
        c['inv'].set(m.invert)
    
    def apply_blend_mode(self, a, b, mode, mix_a, mix_b):
        if mode == "normal" or mode == "add":
            return a * mix_a + b * mix_b
        elif mode == "multiply":
            combined = a * mix_a + b * mix_b
            return combined * 4
        elif mode == "screen":
            combined = a * mix_a + b * mix_b
            return 1 - (1 - combined) * (1 - combined)
        elif mode == "overlay":
            combined = a * mix_a + b * mix_b
            return np.where(combined < 0.5, 2 * combined * combined, 1 - 2 * (1 - combined) * (1 - combined))
        elif mode == "difference":
            return np.abs(a * mix_a - b * mix_b)
        elif mode == "exclusion":
            combined = a * mix_a + b * mix_b
            return combined - 2 * (a * mix_a) * (b * mix_b)
        elif mode == "hard_light":
            combined = a * mix_a + b * mix_b
            return np.where(combined < 0.5, 2 * combined * combined, 1 - 2 * (1 - combined) * (1 - combined))
        elif mode == "soft_light":
            combined = a * mix_a + b * mix_b
            return (1 - 2 * combined) * combined * combined + 2 * combined * combined
        elif mode == "color_dodge":
            combined = a * mix_a + b * mix_b
            return np.minimum(1, combined / (1 - combined + 0.001))
        elif mode == "color_burn":
            combined = a * mix_a + b * mix_b
            return 1 - np.minimum(1, (1 - combined) / (combined + 0.001))
        elif mode == "darken":
            return np.minimum(a * mix_a, b * mix_b)
        elif mode == "lighten":
            return np.maximum(a * mix_a, b * mix_b)
        elif mode == "linear_light":
            combined = a * mix_a + b * mix_b
            return np.clip(combined * 2 - 0.5, 0, 1)
        elif mode == "pin_light":
            am, bm = a * mix_a, b * mix_b
            return np.where(bm < 0.5, np.minimum(am, 2 * bm), np.maximum(am, 2 * bm - 1))
        elif mode == "vivid_light":
            combined = a * mix_a + b * mix_b
            return np.where(combined < 0.5, 1 - (1 - combined) / (2 * combined + 0.001), combined / (2 * (1 - combined) + 0.001))
        else:
            return a * mix_a + b * mix_b
    
    def blend_frames(self, fa, oa, fb, ob, bp):
        mix = max(0.0, min(1.0, self.mix + self.mix_mod.get_value(bp) * 0.5))
        if fa is None:
            fa = np.zeros_like(self.blend_buffer)
            oa = 1.0
        if fb is None:
            fb = np.zeros_like(self.blend_buffer)
            ob = 1.0
        mix_a = oa * (1.0 - mix)
        mix_b = ob * mix
        self.blend_buffer[:] = self.apply_blend_mode(fa, fb, self.blend_mode, mix_a, mix_b)
        mb = self.mbright.get()
        mc = self.mcontr.get()
        if mb != 0 or mc != 1.0:
            self.blend_buffer += mb
            self.blend_buffer -= 0.5
            self.blend_buffer *= mc
            self.blend_buffer += 0.5
        np.clip(self.blend_buffer, 0, 1, out=self.blend_buffer)
        np.multiply(self.blend_buffer, 255, out=self.blend_buffer)
        self.output_buffer[:] = self.blend_buffer.astype(np.uint8)
        return self.output_buffer
    
    def sync_start(self):
        self.sync_var.set("Syncing...")
        self.root.update()
        self.metronome.set_bpm(self.bpm)
        self.metronome.set_beats_per_bar(self.beats_per_bar)
        self.metronome.prepare_start()
        self.channel_a.reset_position()
        self.channel_b.reset_position()
        self.audio_track.stop()
        self.loop_trigger_flag = False
        
        # Calculate start offset in milliseconds based on global_loop_start (now in beats)
        beat_duration_ms = (60.0 / self.bpm) * 1000.0
        loop_start_ms = int(self.global_loop_start * beat_duration_ms)
        
        audio_started = False
        if self.audio_track.enabled and self.audio_track.path:
            # Start audio at the loop_start position, not 0
            self.audio_track.play(loop_start_ms)
            for _ in range(20):
                time.sleep(0.01)
                if self.audio_track.is_active():
                    audio_started = True
                    break
        
        self.start_time = time.perf_counter()
        self.last_update_time = self.start_time
        # Start beat position at global_loop_start beats (already in beats now)
        self.beat_position = self.global_loop_start
        
        self.metronome.synchronized_start(self.start_time)
        self.processor.start_proc(self.start_time)
        self.sync_ready = True
        self.sync_var.set("SYNC")
        
    def toggle_play(self):
        if self.playing:
            self.playing = False
            self.metronome.stop()
            self.processor.stop_proc()
            self.audio_track.stop()
            self.play_btn.config(text="Play")
            self.sync_var.set("")
            self.sync_ready = False
        else:
            self.playing = True
            self.play_btn.config(text="Pause")
            if not self.processor.is_alive():
                self.processor = VideoProcessor(self)
            self.sync_start()
            
    def stop(self):
        self.playing = False
        self.metronome.stop()
        self.processor.stop_proc()
        self.audio_track.stop()
        self.play_btn.config(text="Play")
        self.sync_var.set("")
        self.sync_ready = False
        
    def rewind(self):
        was = self.playing
        if was:
            self.metronome.stop()
            self.processor.stop_proc()
            self.audio_track.stop()
        self.beat_position = 0
        self.channel_a.reset_position()
        self.channel_b.reset_position()
        if was:
            if not self.processor.is_alive():
                self.processor = VideoProcessor(self)
            self.sync_start()
    
    def update_loop(self):
        fd = self.processor.get_frame()
        if fd:
            blended, bp = fd
            self.beat_position = bp
            self.beat_var.set(f"Beat: {int(round(bp))}")
            self.beat_flash.configure(bg="white" if bp % 1 < 0.1 else "gray")
            rgb = blended[:, :, ::-1]
            img = Image.fromarray(rgb)
            self.preview_photo = ImageTk.PhotoImage(img)
            self.preview_canvas.create_image(0, 0, anchor=tk.NW, image=self.preview_photo)
        
        # Update timeline playhead
        if hasattr(self, 'timeline_widget'):
            self.timeline_widget.update_playhead()
        
        now = time.perf_counter()
        dt = now - self.last_update_time
        if dt > 0:
            self.fps_var.set(f"FPS:{1/dt:.0f}")
        self.last_update_time = now
        self.root.after(8, self.update_loop)
        
    def export_video(self):
        if not self.channel_a.cap and not self.channel_b.cap:
            messagebox.showerror("Error", "Load a video first")
            return
        dlg = ExportDialog(self.root, self.bpm, self.beats_per_bar)
        if not dlg.result:
            return
        fmt = dlg.result.get('format', 'avi')
        ext = '.mov' if fmt == 'mov' else '.mp4' if fmt == 'mp4' else '.avi'
        p = filedialog.asksaveasfilename(defaultextension=ext, filetypes=[("MOV", "*.mov"), ("MP4", "*.mp4"), ("AVI", "*.avi")])
        if not p:
            return
        self.status.set("Exporting...")
        threading.Thread(target=self.do_export, args=(p, dlg.result), daemon=True).start()

    def do_export(self, path, s):
        try:
            bars, fps, w, h = s['bars'], s['fps'], s['width'], s['height']
            fmt = s.get('format', 'avi')
            self.channel_a.set_target_size(w, h)
            self.channel_b.set_target_size(w, h)
            
            dur = bars * self.beats_per_bar * 60 / self.bpm
            tot = int(dur * fps)
            ext = os.path.splitext(path)[1].lower()
            if ext == '.avi' or fmt == 'avi':
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                path = os.path.splitext(path)[0] + '.avi'
            elif ext == '.mov' or fmt == 'mov':
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                path = os.path.splitext(path)[0] + '.mov'
            else:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                path = os.path.splitext(path)[0] + '.mp4'
            out = cv2.VideoWriter(path, fourcc, fps, (w, h))
            if not out.isOpened():
                raise Exception("Failed to create video writer")
            self.channel_a.reset_position()
            self.channel_b.reset_position()
            ft = 1 / fps
            
            for i in range(tot):
                bp = (i / fps) * self.bpm / 60
                fa, oa, fb, ob = None, 1.0, None, 1.0
                
                r = self.channel_a.get_frame(bp, ft, self.bpm, self.beats_per_bar)
                if r:
                    fa, oa = r
                r = self.channel_b.get_frame(bp, ft, self.bpm, self.beats_per_bar)
                if r:
                    fb, ob = r
                
                mix = max(0.0, min(1.0, self.mix + self.mix_mod.get_value(bp) * 0.5))
                
                if fa is None:
                    fa = np.zeros((h, w, 3), dtype=np.float32)
                    oa = 1.0
                if fb is None:
                    fb = np.zeros((h, w, 3), dtype=np.float32)
                    ob = 1.0
                
                mix_a = oa * (1.0 - mix)
                mix_b = ob * mix
                
                blended = self.apply_blend_mode(fa, fb, self.blend_mode, mix_a, mix_b)
                
                mb = self.mbright.get()
                mc = self.mcontr.get()
                if mb != 0 or mc != 1.0:
                    blended = blended + mb
                    blended = (blended - 0.5) * mc + 0.5
                
                frame_out = (np.clip(blended, 0, 1) * 255).astype(np.uint8)
                out.write(frame_out)
                
                if i % 30 == 0:
                    pct = i / tot * 100
                    self.root.after(0, lambda p=pct: self.status.set(f"Export: {p:.0f}%"))
            
            out.release()
            self.channel_a.set_target_size(self.preview_width, self.preview_height)
            self.channel_b.set_target_size(self.preview_width, self.preview_height)
            self.root.after(0, lambda: self.status.set(f"Done: {os.path.basename(path)}"))
            self.root.after(0, lambda: messagebox.showinfo("Done", f"Exported {path}\n{dur:.1f}s"))
        except Exception as e:
            self.channel_a.set_target_size(self.preview_width, self.preview_height)
            self.channel_b.set_target_size(self.preview_width, self.preview_height)
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
            
    def save_preset(self):
        p = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if not p:
            return
        d = {'bpm': self.bpm, 'bpb': self.beats_per_bar, 'mix': self.mix, 'blend': self.blend_mode,
             'mbright': self.mbright.get(), 'mcontr': self.mcontr.get(), 'mix_mod': self.mix_mod.to_dict(),
             'ch_a': self.channel_a.to_dict(), 'ch_b': self.channel_b.to_dict(),
             'metro': self.metro_var.get(), 'mvol': self.mvol.get(),
             'gloop_en': self.gloop_en.get(), 'gloop_start': self.gloop_start.get(), 'gloop_end': self.gloop_end.get(),
             'audio_en': self.audio_en.get(), 'latency': self.latency_ms.get()}
        with open(p, 'w') as f:
            json.dump(d, f, indent=2)
        self.status.set(f"Saved preset")
    
    def load_preset(self):
        p = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not p:
            return
        with open(p, 'r') as f:
            d = json.load(f)
        self.bpm = d.get('bpm', 120)
        self.bpm_var.set(self.bpm)
        self.metronome.set_bpm(self.bpm)
        self.beats_per_bar = d.get('bpb', 4)
        self.bpb_var.set(self.beats_per_bar)
        self.mix = d.get('mix', 0.5)
        self.mix_var.set(self.mix)
        self.blend_mode = d.get('blend', 'normal')
        self.blend_var.set(self.blend_mode)
        self.mbright.set(d.get('mbright', 0))
        self.mcontr.set(d.get('mcontr', 1))
        if 'metro' in d:
            self.metro_var.set(d['metro'])
            self.metronome.enabled = self.metro_var.get()
        if 'mvol' in d:
            self.mvol.set(d['mvol'])
            self.metronome.update_volume(self.mvol.get())
        if 'gloop_en' in d:
            self.gloop_en.set(d['gloop_en'])
            self.global_loop_enabled = self.gloop_en.get()
        if 'gloop_start' in d:
            self.gloop_start.set(d['gloop_start'])
            self.global_loop_start = self.gloop_start.get()
        if 'gloop_end' in d:
            self.gloop_end.set(d['gloop_end'])
            self.global_loop_end = self.gloop_end.get()
        if 'audio_en' in d:
            self.audio_en.set(d['audio_en'])
            self.audio_track.enabled = self.audio_en.get()
        if 'latency' in d:
            self.latency_ms.set(d['latency'])
        if 'mix_mod' in d:
            self.mix_mod.from_dict(d['mix_mod'])
            self.update_mod_ui(self.mix_mod_c, self.mix_mod)
        if 'ch_a' in d:
            self.channel_a.from_dict(d['ch_a'])
            self.update_ch_ui(self.channel_a, self.ch_a)
        if 'ch_b' in d:
            self.channel_b.from_dict(d['ch_b'])
            self.update_ch_ui(self.channel_b, self.ch_b)
        self.status.set("Loaded preset")
    
    def save_project(self):
        p = filedialog.asksaveasfilename(defaultextension=".vmproj", filetypes=[("Project", "*.vmproj")])
        if not p:
            return
        d = {'bpm': self.bpm, 'bpb': self.beats_per_bar, 'mix': self.mix, 'blend': self.blend_mode,
             'mbright': self.mbright.get(), 'mcontr': self.mcontr.get(), 'metro': self.metro_var.get(), 'mvol': self.mvol.get(),
             'mix_mod': self.mix_mod.to_dict(), 'ch_a': self.channel_a.to_dict(True), 'ch_b': self.channel_b.to_dict(True),
             'gloop_en': self.gloop_en.get(), 'gloop_start': self.gloop_start.get(), 'gloop_end': self.gloop_end.get(),
             'audio_en': self.audio_en.get(), 'latency': self.latency_ms.get(),
             'audio_path': self.audio_track.path if hasattr(self.audio_track, 'path') and self.audio_track.path else None}
        with open(p, 'w') as f:
            json.dump(d, f, indent=2)
        self.status.set("Saved project")
    
    def load_project(self):
        p = filedialog.askopenfilename(filetypes=[("Project", "*.vmproj *.json")])
        if not p:
            return
        self.stop()
        with open(p, 'r') as f:
            d = json.load(f)
        self.bpm = d.get('bpm', 120)
        self.bpm_var.set(self.bpm)
        self.metronome.set_bpm(self.bpm)
        self.beats_per_bar = d.get('bpb', 4)
        self.bpb_var.set(self.beats_per_bar)
        self.metronome.set_beats_per_bar(self.beats_per_bar)
        self.mix = d.get('mix', 0.5)
        self.mix_var.set(self.mix)
        self.blend_mode = d.get('blend', 'normal')
        self.blend_var.set(self.blend_mode)
        self.mbright.set(d.get('mbright', 0))
        self.mcontr.set(d.get('mcontr', 1))
        self.metro_var.set(d.get('metro', False))
        self.metronome.enabled = self.metro_var.get()
        self.mvol.set(d.get('mvol', 0.5))
        self.metronome.update_volume(self.mvol.get())
        if 'gloop_en' in d:
            self.gloop_en.set(d['gloop_en'])
            self.global_loop_enabled = self.gloop_en.get()
        if 'gloop_start' in d:
            self.gloop_start.set(d['gloop_start'])
            self.global_loop_start = self.gloop_start.get()
        if 'gloop_end' in d:
            self.gloop_end.set(d['gloop_end'])
            self.global_loop_end = self.gloop_end.get()
        if 'audio_en' in d:
            self.audio_en.set(d['audio_en'])
            self.audio_track.enabled = self.audio_en.get()
        if 'latency' in d:
            self.latency_ms.set(d['latency'])
        if 'audio_path' in d and d['audio_path']:
            audio_path = d['audio_path']
            if os.path.exists(audio_path):
                if hasattr(self, 'timeline_widget') and self.timeline_widget:
                    self.timeline_widget.load_audio_waveform(audio_path)
                if self.audio_track.load(audio_path):
                    self.audio_status.set(os.path.basename(audio_path)[:10])
                    self.status.set(f"Loaded project with audio")
                else:
                    self.status.set(f"Loaded project (audio load failed)")
            else:
                self.status.set(f"Loaded project (audio file not found)")
        else:
            self.status.set("Loaded project")
        if 'mix_mod' in d:
            self.mix_mod.from_dict(d['mix_mod'])
            self.update_mod_ui(self.mix_mod_c, self.mix_mod)
        if 'ch_a' in d:
            self.channel_a.from_dict(d['ch_a'], True)
            self.update_ch_ui(self.channel_a, self.ch_a)
        if 'ch_b' in d:
            self.channel_b.from_dict(d['ch_b'], True)
            self.update_ch_ui(self.channel_b, self.ch_b)


class ExportDialog:
    def __init__(self, parent, bpm, bpb):
        self.result = None
        self.bpm = bpm
        self.bpb = bpb
        
        # Define aspect ratios and their resolutions
        self.aspect_ratios = {
            "16:9": ["1280x720", "1920x1080", "2560x1440", "3840x2160"],
            "4:3": ["640x480", "800x600", "1024x768", "1600x1200"],
            "1:1": ["720x720", "1080x1080", "1440x1440", "2160x2160"],
            "3:4": ["480x640", "600x800", "768x1024", "1200x1600"],
            "9:16": ["720x1280", "1080x1920", "1440x2560", "2160x3840"]
        }
        
        dlg = tk.Toplevel(parent)
        dlg.title("Export Video")
        dlg.geometry("300x330")
        dlg.transient(parent)
        dlg.grab_set()
        
        f = ttk.Frame(dlg, padding="15")
        f.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(f, text="Bars:").pack(anchor=tk.W)
        row1 = ttk.Frame(f)
        row1.pack(fill=tk.X, pady=(0, 8))
        self.bars = tk.IntVar(value=8)
        ttk.Spinbox(row1, from_=1, to=999, textvariable=self.bars, width=6, command=self.update_duration).pack(side=tk.LEFT)
        self.dur_var = tk.StringVar()
        self.update_duration()
        ttk.Label(row1, textvariable=self.dur_var).pack(side=tk.LEFT, padx=10)
        
        ttk.Label(f, text="Format:").pack(anchor=tk.W)
        row2 = ttk.Frame(f)
        row2.pack(fill=tk.X, pady=(0, 8))
        self.format = tk.StringVar(value="avi")
        ttk.Radiobutton(row2, text="MOV", variable=self.format, value="mov").pack(side=tk.LEFT)
        ttk.Radiobutton(row2, text="MP4", variable=self.format, value="mp4").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(row2, text="AVI", variable=self.format, value="avi").pack(side=tk.LEFT)
        
        ttk.Label(f, text="FPS:").pack(anchor=tk.W)
        self.fps = tk.IntVar(value=24)
        ttk.Combobox(f, textvariable=self.fps, values=[24, 25, 30, 60], width=6).pack(anchor=tk.W, pady=(0, 8))
        
        # Aspect Ratio selector
        ttk.Label(f, text="Aspect Ratio:").pack(anchor=tk.W)
        self.aspect_ratio = tk.StringVar(value="16:9")
        self.aspect_combo = ttk.Combobox(f, textvariable=self.aspect_ratio, 
                                         values=list(self.aspect_ratios.keys()), 
                                         width=12, state="readonly")
        self.aspect_combo.pack(anchor=tk.W, pady=(0, 8))
        self.aspect_combo.bind("<<ComboboxSelected>>", self.update_resolutions)
        
        ttk.Label(f, text="Resolution:").pack(anchor=tk.W)
        self.res = tk.StringVar(value="1920x1080")
        self.res_combo = ttk.Combobox(f, textvariable=self.res, width=12)
        self.res_combo.pack(anchor=tk.W, pady=(0, 8))
        self.update_resolutions()  # Initialize resolution options
        
        ttk.Label(f, text=f"BPM: {bpm}  Beats/Bar: {bpb}").pack(anchor=tk.W, pady=(0, 8))
        
        row_btn = ttk.Frame(f)
        row_btn.pack(fill=tk.X, pady=5)
        tk.Button(row_btn, text="Export", command=lambda: self.on_ok(dlg), width=10).pack(side=tk.LEFT, padx=5)
        tk.Button(row_btn, text="Cancel", command=dlg.destroy, width=10).pack(side=tk.LEFT, padx=5)
        
        dlg.wait_window()
    
    def update_duration(self):
        try:
            dur = self.bars.get() * self.bpb * 60 / self.bpm
            self.dur_var.set(f"= {dur:.1f}s")
        except:
            pass
    
    def update_resolutions(self, event=None):
        """Update resolution options based on selected aspect ratio"""
        aspect = self.aspect_ratio.get()
        resolutions = self.aspect_ratios.get(aspect, ["1920x1080"])
        self.res_combo['values'] = resolutions
        # Set to first resolution if current is not in the new list
        if self.res.get() not in resolutions:
            self.res.set(resolutions[0])
    
    def on_ok(self, dlg):
        r = self.res.get().split('x')
        self.result = {
            'bars': self.bars.get(),
            'fps': self.fps.get(),
            'width': int(r[0]),
            'height': int(r[1]),
            'format': self.format.get()
        }
        dlg.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = VideoMixer(root)
    root.mainloop()
