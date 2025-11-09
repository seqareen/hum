#!/usr/bin/env python3
"""
Ultra Vocal to Humming Converter
Synthesizes pure humming from vocal pitch contour
"""

import numpy as np
import librosa
import soundfile as sf
from scipy import signal
import warnings
warnings.filterwarnings('ignore')


class UltraHummingConverter:
    """Converts vocals to pure therapeutic humming via synthesis"""

    def __init__(self, target_freq=130, sr=44100):
        self.target_freq = target_freq
        self.sr = sr

    def extract_pitch_contour(self, y):
        """Extract pitch and amplitude envelope from vocals"""
        print("Extracting pitch contour...")

        # Use pYIN for better pitch tracking
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('C2'),  # ~65 Hz
            fmax=librosa.note_to_hz('C6'),  # ~1046 Hz
            sr=self.sr,
            frame_length=2048
        )

        # Get amplitude envelope
        hop_length = 512
        S = np.abs(librosa.stft(y, hop_length=hop_length))
        rms = librosa.feature.rms(S=S, hop_length=hop_length)[0]

        # Interpolate pitch for unvoiced segments
        voiced_indices = np.where(voiced_flag)[0]
        if len(voiced_indices) > 0:
            # Fill NaN values with interpolation
            valid_f0 = f0[voiced_flag]
            interp_func = np.interp(
                np.arange(len(f0)),
                voiced_indices,
                valid_f0
            )
            f0_filled = interp_func
        else:
            # If no pitch detected, use target frequency
            f0_filled = np.full(len(f0), self.target_freq)

        return f0_filled, rms, voiced_flag

    def synthesize_humming(self, f0_contour, rms, voiced_flag):
        """Synthesize pure humming tone from pitch contour"""
        print("Synthesizing humming tone...")

        hop_length = 512
        n_frames = len(f0_contour)

        # Shift all pitches to target frequency while preserving contour shape
        # Calculate relative pitch variation
        median_pitch = np.median(f0_contour[voiced_flag]) if np.any(voiced_flag) else self.target_freq

        # Scale pitch variations to be subtle (humming has less pitch variation)
        pitch_deviation = (f0_contour - median_pitch) * 0.3  # Reduce variation by 70%
        target_contour = self.target_freq + pitch_deviation

        # Clip to reasonable range around target
        target_contour = np.clip(target_contour, self.target_freq - 20, self.target_freq + 20)

        # Generate time array
        samples_per_frame = hop_length
        total_samples = n_frames * samples_per_frame

        # Synthesize using harmonic additive synthesis
        y_synth = np.zeros(total_samples)
        t = np.arange(total_samples) / self.sr

        # Generate smooth pitch curve
        frame_times = np.arange(n_frames) * hop_length / self.sr
        pitch_curve = np.interp(t, frame_times, target_contour)

        # Generate amplitude curve
        amp_curve = np.interp(t, frame_times, rms)
        amp_curve = signal.savgol_filter(amp_curve, window_length=2001, polyorder=3)

        # Synthesize fundamental + harmonics (like real humming)
        # Fundamental (130 Hz)
        phase = 2 * np.pi * np.cumsum(pitch_curve) / self.sr
        y_synth += np.sin(phase) * 1.0  # Fundamental is strongest

        # 2nd harmonic (260 Hz) - weaker
        y_synth += np.sin(2 * phase) * 0.4

        # 3rd harmonic (390 Hz) - much weaker
        y_synth += np.sin(3 * phase) * 0.15

        # 4th harmonic (520 Hz) - very weak
        y_synth += np.sin(4 * phase) * 0.05

        # Apply amplitude envelope
        y_synth *= amp_curve

        # Add slight vibrato (natural humming characteristic)
        vibrato_rate = 5.5  # Hz
        vibrato_depth = 0.5  # Hz
        vibrato = np.sin(2 * np.pi * vibrato_rate * t) * vibrato_depth
        vibrato_phase = 2 * np.pi * np.cumsum(vibrato) / self.sr
        y_synth *= (1 + 0.02 * np.sin(vibrato_phase))  # Very subtle

        # Normalize
        y_synth = y_synth / (np.max(np.abs(y_synth)) + 1e-8)

        return y_synth

    def add_nasal_resonance(self, y):
        """Add nasal quality characteristic of humming"""
        print("Adding nasal resonance...")

        # Nasal formants are typically around 250 Hz and 2500 Hz
        # Apply formant filtering

        # Low nasal formant (~250 Hz)
        sos1 = signal.butter(2, [200, 300], 'bandpass', fs=self.sr, output='sos')
        nasal_low = signal.sosfilt(sos1, y) * 0.3

        # High nasal formant (~2500 Hz)
        sos2 = signal.butter(2, [2400, 2600], 'bandpass', fs=self.sr, output='sos')
        nasal_high = signal.sosfilt(sos2, y) * 0.1

        # Mix with original
        y_nasal = y + nasal_low + nasal_high

        # Normalize
        y_nasal = y_nasal / (np.max(np.abs(y_nasal)) + 1e-8)

        return y_nasal

    def apply_final_processing(self, y):
        """Apply final smoothing and quality enhancements"""
        print("Applying final processing...")

        # Very aggressive low-pass to ensure pure tone
        # Cutoff at 1500 Hz (well above 130 Hz fundamental + a few harmonics)
        sos_lp = signal.butter(8, 1500, 'lowpass', fs=self.sr, output='sos')
        y = signal.sosfilt(sos_lp, y)

        # High-pass to remove DC offset and subsonic
        sos_hp = signal.butter(4, 60, 'highpass', fs=self.sr, output='sos')
        y = signal.sosfilt(sos_hp, y)

        # Smooth amplitude envelope to remove any remaining transients
        analytic = signal.hilbert(y)
        envelope = np.abs(analytic)

        # Very smooth envelope (100ms window)
        window_samples = int(self.sr * 0.1)
        kernel = np.ones(window_samples) / window_samples
        smooth_env = np.convolve(envelope, kernel, mode='same')

        # Apply smoothed envelope
        phase = np.angle(analytic)
        y = smooth_env * np.cos(phase)

        # Gentle compression for consistent loudness
        threshold = 0.3
        ratio = 3.0
        mask = np.abs(y) > threshold
        y[mask] = np.sign(y[mask]) * (threshold + (np.abs(y[mask]) - threshold) / ratio)

        # Final normalization to -3dB
        y = y / (np.max(np.abs(y)) + 1e-8) * 0.7

        return y

    def convert(self, input_file, output_file, verbose=True):
        """Main conversion function"""

        if verbose:
            print("="*70)
            print("ULTRA HUMMING CONVERTER")
            print("="*70)
            print(f"\nLoading: {input_file}")

        # Load audio
        y, sr = librosa.load(input_file, sr=self.sr, mono=True)

        if verbose:
            print(f"Duration: {len(y) / sr:.2f}s | Sample Rate: {sr} Hz\n")

        # Extract pitch and amplitude
        f0_contour, rms, voiced_flag = self.extract_pitch_contour(y)

        # Synthesize pure humming
        y_humming = self.synthesize_humming(f0_contour, rms, voiced_flag)

        # Add nasal quality
        y_humming = self.add_nasal_resonance(y_humming)

        # Final processing
        y_final = self.apply_final_processing(y_humming)

        # Save
        sf.write(output_file, y_final, sr)

        if verbose:
            print(f"\nSaved: {output_file}")
            print("="*70)
            print("\nThis version synthesizes PURE humming from your pitch contour")
            print(f"Target frequency: {self.target_freq} Hz")
            print("="*70)


def main():
    import sys

    if len(sys.argv) < 3:
        print("Usage: python ultra_vocal_to_humming.py <input> <output>")
        print("Example: python ultra_vocal_to_humming.py vocal.wav humming.wav")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    converter = UltraHummingConverter(target_freq=130, sr=44100)
    converter.convert(input_file, output_file, verbose=True)


if __name__ == "__main__":
    main()
