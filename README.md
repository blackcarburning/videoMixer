# Video Mixer - Timeline Widget Enhancement

## New Features

### 1. Beat-Based Loop Markers (Updated)

The loop markers now operate on **beats** instead of bars, providing finer control over loop regions. The audio correctly starts at the `Loop Start` position (in beats) when the "Play" button is pressed.

**Changes:**
- Loop markers now use beats as the unit instead of bars
- `sync_start()` method calculates the start offset in milliseconds based on `global_loop_start` in beats
- Audio playback starts at the calculated offset: `loop_start_ms = int(global_loop_start * beat_duration_ms)`
- Beat position is initialized to the loop start: `beat_position = global_loop_start`
- Beat counter displays whole numbers instead of decimals for clearer feedback

### 2. Timeline Widget with Waveform Visualization

A new interactive timeline widget has been added to the UI that displays:

- **Waveform**: Visual representation of the loaded audio track
- **Grid**: Beat markers with emphasis on bar boundaries based on the current BPM and Beats Per Bar settings
- **Loop Points**: Green line for Loop Start and red line for Loop End (now in beats)
- **Playhead**: Yellow line showing the current playback position (clickable for seeking)
- **Loop Region**: Semi-transparent green highlight between loop points

**Features:**
- Automatically loads waveform when audio is loaded
- Updates grid when BPM changes
- Updates playhead position during playback
- **Click anywhere on the timeline to seek to that beat position**

### 3. Interactive Loop Point Editing

You can now drag the Loop Start and Loop End handles directly on the timeline:

- **Click and drag** the green line to move the Loop Start point
- **Click and drag** the red line to move the Loop End point
- **Snap to beats**: Handles automatically snap to the nearest beat
- Changes to handles update the Loop Start/End spinbox values in real-time
- **Click on the timeline** (not on handles) to seek the playhead to that position

### 4. Improved Synchronization

- Audio starts precisely at the calculated time offset corresponding to `global_loop_start` (in beats)
- Visual beat counter displays whole numbers and aligns with the audio from the first frame of playback
- Loop behavior during playback remains unchanged and functional
- Grid and snapping work on beat boundaries for precise control

## Technical Details

### TimelineWidget Class

A new `TimelineWidget` class (inheriting from `tk.Canvas`) has been added with the following methods:

- `load_audio_waveform(audio_path)`: Extracts waveform data using `pygame.sndarray`
- `draw_waveform(w, h)`: Renders the audio waveform
- `draw_grid(w, h)`: Draws beat markers (with emphasis on bars)
- `draw_loop_handles(w, h)`: Renders loop start/end handles (beat-based)
- `draw_playhead(w, h)`: Shows current playback position (beat-based)
- `update_playhead()`: Updates playhead during playback
- `on_mouse_down(event)`: Handles mouse clicks on handles and timeline seeking
- `on_mouse_drag(event)`: Implements drag-and-snap behavior for loop handles
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
   - Use the spinboxes (now in beats, range 0-400), or
   - Drag the handles on the timeline (snaps to beats), or
   - Click anywhere on the timeline to seek the playhead
4. **Play**: Press "Play" - audio will start at the Loop Start position

## Files Modified

- `video_mixer.py`: Main application file
  - Updated `TimelineWidget` class to use beats instead of bars
  - Modified `sync_start()` to work with beat-based loop markers
  - Updated UI labels from "Loop Bars" to "Loop Beats"
  - Changed beat counter to display whole numbers
  - Added clickable seeking on timeline
  - Updated all loop calculations to use beats directly

## Notes

- The waveform is downsampled to ~2000 points for efficient rendering
- Mono and stereo audio files are both supported (stereo is converted to mono for visualization)
- The timeline automatically resizes when the window is resized
- Loop markers are stored in beats, providing finer control than the previous bar-based system
- Default loop is 16 beats (equivalent to 4 bars at 4 beats per bar)
