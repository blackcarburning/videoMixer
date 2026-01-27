# Video Mixer - Timeline Widget Enhancement

## New Features

### 1. Fixed Audio Loop Start Behavior

The audio now correctly starts at the `Loop Start` position (in bars) when the "Play" button is pressed, instead of always starting from the beginning (0). The visual beat counter is also synchronized with this start offset.

**Changes:**
- `sync_start()` method now calculates the start offset in milliseconds based on `global_loop_start`
- Audio playback starts at the calculated offset: `loop_start_ms = int(global_loop_start * bar_duration_ms)`
- Beat position is initialized to the loop start: `beat_position = global_loop_start * beats_per_bar`

### 2. Timeline Widget with Waveform Visualization

A new interactive timeline widget has been added to the UI that displays:

- **Waveform**: Visual representation of the loaded audio track
- **Grid**: Bar markers based on the current BPM and Beats Per Bar settings
- **Loop Points**: Green line for Loop Start and red line for Loop End
- **Playhead**: Yellow line showing the current playback position
- **Loop Region**: Semi-transparent green highlight between loop points

**Features:**
- Automatically loads waveform when audio is loaded
- Updates grid when BPM changes
- Updates playhead position during playback

### 3. Interactive Loop Point Editing

You can now drag the Loop Start and Loop End handles directly on the timeline:

- **Click and drag** the green line to move the Loop Start point
- **Click and drag** the red line to move the Loop End point
- **Snap to bars**: Handles automatically snap to the nearest bar line
- Changes to handles update the Loop Start/End spinbox values in real-time

### 4. Improved Synchronization

- Audio starts precisely at the calculated time offset corresponding to `global_loop_start`
- Visual beat counter aligns with the audio from the first frame of playback
- Loop behavior during playback remains unchanged and functional

## Technical Details

### TimelineWidget Class

A new `TimelineWidget` class (inheriting from `tk.Canvas`) has been added with the following methods:

- `load_audio_waveform(audio_path)`: Extracts waveform data using `pygame.sndarray`
- `draw_waveform(w, h)`: Renders the audio waveform
- `draw_grid(w, h)`: Draws bar markers
- `draw_loop_handles(w, h)`: Renders loop start/end handles
- `draw_playhead(w, h)`: Shows current playback position
- `update_playhead()`: Updates playhead during playback
- `on_mouse_down(event)`: Handles mouse clicks on handles
- `on_mouse_drag(event)`: Implements drag-and-snap behavior
- `on_mouse_up(event)`: Completes the drag operation

### Dependencies

The application requires:
- `tkinter` (built-in with Python on most systems)
- `opencv-python` (cv2)
- `numpy`
- `Pillow` (PIL)
- `pygame` (includes pygame.sndarray for waveform extraction)

Install dependencies with:
```bash
pip install -r requirements.txt
```

## Usage

1. **Load Audio**: Click "Load" in the Transport section to load an audio file
2. **Set BPM**: Adjust the BPM to match your audio track (or use "Tap" button)
3. **Adjust Loop Points**: 
   - Use the spinboxes, or
   - Drag the handles on the timeline
4. **Play**: Press "Play" - audio will start at the Loop Start position

## Files Modified

- `video_mixer.py`: Main application file
  - Added `TimelineWidget` class (lines 673-962)
  - Modified `sync_start()` to respect loop start (lines 1699-1732)
  - Integrated timeline widget into UI (lines 1294-1298)
  - Added timeline updates in `on_bpm()`, `load_audio()`, and `update_loop()`

## Notes

- The waveform is downsampled to ~2000 points for efficient rendering
- Mono and stereo audio files are both supported (stereo is converted to mono for visualization)
- The timeline automatically resizes when the window is resized
