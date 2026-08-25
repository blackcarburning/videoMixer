import numpy as np
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import video_mixer as vm


def test_modulator_basic_waves_and_rand_determinism():
    m = vm.Modulator()
    m.enabled = True
    m.depth = 1.0
    m.rate = 1.0

    m.wave_type = "square"
    assert m.get_value(0.1) == 1.0
    assert m.get_value(0.6) == -1.0

    m.wave_type = "triangle"
    assert np.isclose(m.get_value(0.0), 0.0)
    assert np.isclose(m.get_value(0.25), 1.0)

    m.wave_type = "saw_forward"
    assert np.isclose(m.get_value(0.0), -1.0)
    assert np.isclose(m.get_value(0.5), 0.0)

    m.wave_type = "rand"
    v1 = m.get_value(4.2)
    v2 = m.get_value(4.8)
    v3 = m.get_value(5.1)
    assert np.isclose(v1, v2)
    assert not np.isclose(v2, v3)


def test_modulator_envelope_and_negative_positions():
    m = vm.Modulator()
    m.enabled = True
    m.depth = 1.0
    m.rate = 1.0
    m.wave_type = "envelope"
    v_start = m.get_value(-0.25, bpm=120.0, env_attack=0.1, env_release=0.1)
    v_mid = m.get_value(0.1, bpm=120.0, env_attack=0.1, env_release=0.1)
    assert -1.0 <= v_start <= 1.0
    assert -1.0 <= v_mid <= 1.0


def test_blend_modes_use_pairwise_math():
    a = np.array([[[0.2, 0.2, 0.2]]], dtype=np.float32)
    b = np.array([[[0.5, 0.5, 0.5]]], dtype=np.float32)

    mul = vm.VideoMixer.apply_blend_mode(None, a, b, "multiply", 1.0, 1.0)
    assert np.allclose(mul, 0.15, atol=1e-5)

    scr = vm.VideoMixer.apply_blend_mode(None, a, b, "screen", 1.0, 1.0)
    assert np.allclose(scr, 0.4, atol=1e-4)

    diff = vm.VideoMixer.apply_blend_mode(None, a, b, "difference", 1.0, 1.0)
    assert np.allclose(diff, 0.25, atol=1e-4)


def test_gate_step_helpers_across_timebases():
    ch = vm.VideoChannel(64, 48)
    ch.gate_timebase = 4.0
    assert ch._get_gate_step(0.0) == 0
    assert ch._get_gate_step(1.0) == 4
    ch.gate_timebase = 2.0
    assert ch._get_gate_step(1.0) == 8
    assert 0.0 <= ch._get_step_position(1.125) < 1.0


def test_channel_serialization_roundtrip_and_missing_keys():
    ch = vm.VideoChannel(64, 48)
    ch.brightness = 0.3
    ch.echo_amount = 0.4
    ch.spin_amount = 0.5
    ch.zoom_mod.enabled = True
    ch.zoom_mod.depth = 0.75
    data = ch.to_dict()

    loaded = vm.VideoChannel(64, 48)
    loaded.from_dict(data, load_video=False)
    assert loaded.brightness == ch.brightness
    assert loaded.echo_amount == ch.echo_amount
    for attr in vm.VideoChannel.MODULATOR_ATTRS:
        assert getattr(loaded, attr).to_dict() == getattr(ch, attr).to_dict()

    older = vm.VideoChannel(64, 48)
    older.from_dict({"brightness": 0.2}, load_video=False)
    assert np.isclose(older.brightness, 0.2)
    assert np.isclose(older.opacity, 1.0)


def test_disintegration_and_mirror_shape_dtype_and_stability():
    ch = vm.VideoChannel(65, 49)
    frame = np.random.rand(49, 65, 3).astype(np.float32)

    for fn in (ch._apply_particle_dissolve, ch._apply_thanos_snap, ch._apply_glitch_dissolve):
        out0 = fn(frame, 0.0, 1.0)
        assert out0.shape == frame.shape
        assert out0.dtype == frame.dtype
        out1 = fn(frame, 1.0, 1.0)
        assert out1.shape == frame.shape
        assert np.isfinite(out1).all()

    ch.mirror_enabled = True
    for mode in ("Horizontal", "Vertical", "Quad", "Kaleido"):
        ch.mirror_mode = mode
        mirrored = ch._apply_mirror(frame, beat_pos=1.0, bpm=120.0, env_attack=0.1, env_release=0.1)
        assert mirrored.shape == frame.shape
