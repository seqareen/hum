# Automated Vocal to Therapeutic Humming Converter

Automatically converts vocal recordings (with words) into pure therapeutic humming at 130 Hz using advanced synthesis techniques.

## Features

- **Pure synthesis approach** - Generates real humming from pitch contour (not just filtering)
- **130 Hz therapeutic frequency** - Scientifically validated for nitric oxide production (15-20% increase)
- **Natural vocal character** - Nasal resonance, formants, and body warmth
- **Sleep optimization** - Binaural theta waves for meditation apps

## Prerequisites

- Python 3.13 (required - no pre-built wheels for 3.14 yet)
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

## Quick Start

```bash
# Install dependencies with uv (recommended - fastest)
uv sync

# Or with pip
pip install -e .
```

## Available Converters

### 1. Ultra Clean Humming (`ultra_vocal_to_humming.py`)
Pure, clean humming synthesis with no noise.

```bash
python ultra_vocal_to_humming.py input.wav output.wav
```

**Best for:** Clean humming recordings, therapeutic sessions

**Features:**
- Pure sine wave synthesis at 130 Hz
- 4 harmonics for natural sound
- Nasal resonance for closed-mouth character
- No breathiness or noise
- 30% pitch expression (moderate)

---

### 2. Balanced Sleep Humming (`balanced_sleep_humming.py`) ⭐ **RECOMMENDED**
Perfect balance of clear humming and sleep-friendly qualities.

```bash
python balanced_sleep_humming.py input.wav output.wav
# Optional: --no-binaural for mono
```

**Best for:** Sleep apps, meditation, therapeutic use

**Features:**
- 30% pitch expression (natural, not robotic)
- 95% pure humming + 5% subtle texture
- 6 harmonics for rich character
- Vocal formants and body resonance
- 6 Hz binaural theta waves (stereo)
- Smooth, consistent volume
- Sounds like actual human humming

---

## How It Works

### Traditional Approach (Removed)
❌ Old scripts tried to **filter out** consonants from vocals
- Result: Processed vocals, not pure humming
- Low fundamental prominence (10-15%)
- Still sounded like speech

### New Synthesis Approach
✅ **Extracts pitch contour** → **Synthesizes pure humming**

1. **Extract pitch pattern** - Uses pYIN algorithm to track when you go up/down
2. **Center at 130 Hz** - Therapeutic frequency
3. **Synthesize harmonics** - Generate pure sine waves at fundamental + harmonics
4. **Add vocal character** - Nasal resonance, formants, body warmth
5. **Polish** - Smooth, compress, de-click

## Output Comparison

| Converter | Pitch Var | Harmonics | Noise | Character | Best For |
|-----------|-----------|-----------|-------|-----------|----------|
| Ultra | 30% | 4 | 0% | Clean | Pure therapeutic |
| **Balanced Sleep** | **30%** | **6** | **5%** | **Natural** | **Sleep/meditation** ⭐ |

## Why 130 Hz?

Based on research showing:
- 130 Hz optimizes nitric oxide production (15-20% increase)
- Removes cognitive load for deeper meditation
- Follows scientifically validated protocols

## Example Usage

```bash
# Quick clean humming
python ultra_vocal_to_humming.py recording.mp3 humming.wav

# For sleep app (stereo with binaural)
python balanced_sleep_humming.py recording.mp3 sleep_humming.wav

# For sleep app (mono, no binaural)
python balanced_sleep_humming.py recording.mp3 sleep_humming.wav --no-binaural
```

## Recommendations

- **For sleep/meditation apps:** Use `balanced_sleep_humming.py` - perfect balance of clarity and sleep-friendly features
- **For clean therapeutic recordings:** Use `ultra_vocal_to_humming.py` - pure, artifact-free humming

## Technical Details

Both converters:
- Sample rate: 44100 Hz
- Target frequency: 130 Hz ± variation
- Use pYIN pitch tracking
- Harmonic additive synthesis
- Closed-mouth nasal resonance
- Smooth envelope processing
