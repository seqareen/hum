# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an automated vocal-to-therapeutic-humming converter that transforms vocal recordings into pure therapeutic humming at 130 Hz. The project uses **synthesis-based approach** where pitch contours are extracted and pure humming is generated from scratch, rather than filtering existing vocals.

## Core Architecture

The project contains two synthesis-based converters with different quality profiles:

### Synthesis Approach (Current)
Both converters follow this pipeline:
1. **Extract pitch contour** using pYIN algorithm (librosa)
2. **Extract amplitude envelope** using RMS
3. **Synthesize pure tones** at 130 Hz using harmonic additive synthesis
4. **Add vocal character** (nasal resonance, formants, body warmth)
5. **Final processing** (smoothing, compression, de-clicking)

This approach produces **actual humming sounds** rather than filtered vocals.

### Available Converters

#### 1. Ultra Clean (`ultra_vocal_to_humming.py`)
- **Purpose**: Pure, clean humming with no artifacts
- **Harmonics**: 4 (fundamental + 3)
- **Pitch variation**: 30% (moderate expression)
- **Special features**: Nasal resonance, gentle vibrato
- **Use case**: Therapeutic sessions, clean recordings

#### 2. Balanced Sleep (`balanced_sleep_humming.py`) ⭐ **RECOMMENDED**
- **Purpose**: Perfect balance for sleep/meditation apps
- **Harmonics**: 6 (rich but not complex)
- **Pitch variation**: 30% (natural, not robotic)
- **Special features**:
  - 95% humming + 5% subtle texture
  - 6 Hz binaural theta waves (stereo)
  - Vocal formants for warmth
  - Consistent volume (important for sleep)
- **Use case**: Sleep apps, meditation, therapeutic use

## Key Technical Concepts

### Pitch Extraction
Both converters use **pYIN algorithm** (librosa.pyin):
- More accurate than basic pitch tracking
- Returns: f0 (frequency), voiced_flag (boolean), voiced_probs
- Parameters: hop_length varies (256-512)
- Unvoiced segments are interpolated (linear or spline)

### Harmonic Synthesis
**Additive synthesis** generates pure tones:
```python
phase = 2 * π * cumsum(pitch_curve) / sr
y = sin(phase) * amplitude_1  # Fundamental
y += sin(2 * phase) * amplitude_2  # 2nd harmonic
# ... more harmonics
```

**Harmonic ratios** define character:
- Strong fundamental (1.0-1.5x) = warm, grounding
- Strong 2nd harmonic (0.6-0.8x) = body, fullness
- Weaker upper harmonics (0.05-0.35x) = brightness, character

### Vocal Character Elements

1. **Nasal Resonance** (closed-mouth humming):
   - Primary: 230-320 Hz
   - Secondary: 1900-2400 Hz
   - Added via bandpass filtering

2. **Formants** (human vocal tract):
   - F1 (~300 Hz): Chest/body resonance
   - F2 (~850 Hz): Presence/clarity
   - F3 (~2300 Hz): Nasal quality

3. **Body Resonance**:
   - 100-170 Hz: Chest cavity
   - 65 Hz: Sub-harmonic for depth (Balanced only)

### Smoothing Strategies

Different converters use different smoothing levels:

- **Light** (Ultra): Savgol filter window=21, 30ms envelope
- **Moderate** (Balanced): Savgol window=21-31, 40-50ms envelope

### Binaural Beats
Balanced Sleep converter adds **6 Hz theta waves**:
- Creates stereo output
- Left channel: original
- Right channel: phase-modulated by 6 Hz
- Brain synchronizes to theta state (relaxation)

## Development Commands

### Installation
```bash
# Using uv (recommended)
uv sync

# Using pip
pip install -e .
```

### Running Converters
```bash
# Ultra clean
python ultra_vocal_to_humming.py input.mp3 output.wav

# Balanced sleep (recommended)
python balanced_sleep_humming.py input.mp3 output.wav
python balanced_sleep_humming.py input.mp3 output.wav --no-binaural  # mono
```

### Testing Different Approaches
Both scripts can be imported as modules:
```python
from balanced_sleep_humming import BalancedSleepHummingConverter

converter = BalancedSleepHummingConverter(target_freq=130, sr=44100)
converter.convert("input.mp3", "output.wav", add_binaural=True)
```

## Key Technical Details

### Python Version Requirement
- **Required**: Python 3.13
- **Reason**: No pre-built wheels for dependencies on Python 3.14+
- Specified in `pyproject.toml` as `requires-python = ">=3.13"`

### Dependencies
- **librosa**: Pitch tracking (pYIN), STFT, audio loading
- **soundfile**: Writing output files
- **scipy**: Filtering (butter, sosfilt), signal processing
- **numpy**: Array operations, synthesis

### Audio Parameters
- **Sample Rate**: 44100 Hz (hardcoded)
- **Target Frequency**: 130 Hz (configurable via `target_freq` parameter)
- **Hop Lengths**: 256-512 samples (varies by converter)
- **FFT Sizes**: 2048 (standard across both)

### File Structure
```
hum/
├── ultra_vocal_to_humming.py           # Clean humming (227 lines)
├── balanced_sleep_humming.py           # Balanced sleep - RECOMMENDED (297 lines)
├── pyproject.toml                      # Dependencies & build config
├── uv.lock                             # Locked dependency versions
├── README.md                           # User documentation
└── CLAUDE.md                           # This file
```

## Modification Guidelines

### Adding New Converters
Follow the established pattern:
1. Create class inheriting from nothing (standalone)
2. Implement `extract_pitch_contour(y)` method
3. Implement `synthesize_*_humming(...)` method with harmonic synthesis
4. Add optional `add_*_character(y)` methods for vocal features
5. Implement `apply_final_processing(y)` method
6. Create `convert(input_file, output_file, ...)` main method
7. Add `main()` function for CLI usage

### Tuning Harmonic Balance
Adjust amplitudes in synthesis method:
```python
# More warmth: Increase fundamental and 2nd harmonic
y += np.sin(phase) * 1.5          # Was 1.0
y += np.sin(2 * phase) * 0.8      # Was 0.5

# More brightness: Increase upper harmonics
y += np.sin(5 * phase) * 0.20     # Was 0.10
```

### Adjusting Expressiveness
Change pitch variation percentage:
```python
# More expressive (follows voice more)
pitch_deviation = (f0_contour - median_pitch) * 0.5  # Was 0.3

# More monotone (drone-like)
pitch_deviation = (f0_contour - median_pitch) * 0.1  # Was 0.3
```

### Tuning Smoothness
Adjust Savgol filter parameters:
```python
# Smoother (less variation)
f0_filled = signal.savgol_filter(f0_filled, window_length=51, polyorder=2)

# More responsive (more variation)
f0_filled = signal.savgol_filter(f0_filled, window_length=11, polyorder=3)
```

### Common Issues

1. **Sounds too robotic**: Increase pitch variation % or reduce smoothing
2. **Sounds unstable**: Increase smoothing window size or reduce pitch variation %
3. **Too thin/weak**: Increase fundamental amplitude or add more harmonics
4. **Too bright**: Reduce upper harmonic amplitudes or lower cutoff frequency
5. **Sounds like airplane**: Reduce noise %, increase pitch variation %, reduce smoothing

## Historical Context

### Original Filtering Approach (Removed)
The project originally used two filtering-based converters:
- `vocal_to_humming.py`: Basic greedy parameter search
- `advanced_vocal_to_humming.py`: Spectral gating with optimization

**Problems**:
- Only 10-15% fundamental energy at 130 Hz
- Still sounded like processed vocals, not humming
- Consonants were reduced but vocal character remained

**Bugs fixed before removal**:
1. STFT dimension mismatch (different n_fft values)
2. Incorrect high_freq_rolloff calculation
3. Inefficient frequency array recreation in loops
4. Pitch detection fallback missing
5. Unicode encoding errors in status output

### Intermediate Converters (Removed)
Three additional converters were created and later removed:
- `warm_vocal_to_humming.py`: Too many harmonics (7), too expressive
- `smooth_vocal_to_humming.py`: Over-smoothed, redundant with ultra
- `sleep_humming_converter.py`: Too drone-like, sounded like airplane engine

**Issues with removed converters**:
- Warm: Subtle breathiness added noise/static feel
- Sleep Drone: Heavy noise masking (25%) drowned out humming character, minimal pitch variation (10%) made it sound mechanical

### Synthesis Approach (Current)
Complete reimplementation using harmonic synthesis:
- Extracts pitch contour, discards original audio
- Generates pure sine waves at 130 Hz + harmonics
- Adds vocal character through filtering
- Results in **actual humming** sound, not filtered vocals

This approach produces 35-40% fundamental energy (vs 10-15% with filtering).

### Final Simplification
Reduced from 5 converters to 2 essential ones:
- **Ultra**: Pure, clean humming (0% noise)
- **Balanced**: Sleep/meditation optimized (5% subtle texture, binaural)

This covers all real-world use cases without redundancy.
