#!/usr/bin/env python3
"""
Balanced Sleep Humming Converter
Perfect balance of humming character and sleep-friendly qualities
"""

import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from scipy.interpolate import UnivariateSpline
import warnings
warnings.filterwarnings('ignore')


class BalancedSleepHummingConverter:
    """Converts vocals to balanced sleep humming - clear humming with subtle noise"""

    def __init__(self, target_freq=130, sr=44100):
        self.target_freq = target_freq
        self.sr = sr

    def extract_pitch_contour(self, y):
        """Extract pitch contour with moderate smoothing"""
        print("Extracting pitch contour...")

        hop_length = 512  # Moderate hop

        # Use pYIN for pitch tracking
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C6'),
            sr=self.sr,
            frame_length=2048,
            hop_length=hop_length
        )

        # Get amplitude envelope
        S = np.abs(librosa.stft(y, hop_length=hop_length, n_fft=2048))
        rms = librosa.feature.rms(S=S, frame_length=2048, hop_length=hop_length)[0]

        # Ensure same length
        min_len = min(len(f0), len(rms))
        f0 = f0[:min_len]
        voiced_flag = voiced_flag[:min_len]
        rms = rms[:min_len]

        # Interpolate pitch
        voiced_indices = np.where(voiced_flag)[0]
        if len(voiced_indices) > 3:
            valid_f0 = f0[voiced_flag]
            # Moderate smoothing spline
            spline = UnivariateSpline(voiced_indices, valid_f0, s=len(voiced_indices)*5, k=3)
            f0_filled = spline(np.arange(len(f0)))
        else:
            f0_filled = np.full(len(f0), self.target_freq)

        # Moderate smoothing - keep expression but smooth out roughness
        if len(f0_filled) > 21:
            f0_filled = signal.savgol_filter(f0_filled, window_length=21, polyorder=3)

        # Smooth amplitude moderately
        if len(rms) > 31:
            rms = signal.savgol_filter(rms, window_length=31, polyorder=3)

        return f0_filled, rms, voiced_flag, hop_length

    def synthesize_balanced_humming(self, f0_contour, rms, voiced_flag, hop_length):
        """Synthesize humming with clear tonal quality"""
        print("Synthesizing balanced humming...")

        n_frames = len(f0_contour)

        # Calculate pitch deviation
        median_pitch = np.median(f0_contour[voiced_flag]) if np.any(voiced_flag) else self.target_freq

        # Keep 30% variation - enough to sound human, not too much to be distracting
        pitch_deviation = (f0_contour - median_pitch) * 0.3
        target_contour = self.target_freq + pitch_deviation

        # Moderate range
        target_contour = np.clip(target_contour, self.target_freq - 20, self.target_freq + 20)

        # Smooth but keep character
        if len(target_contour) > 11:
            target_contour = signal.savgol_filter(target_contour, window_length=11, polyorder=2)

        # Generate time array
        samples_per_frame = hop_length
        total_samples = n_frames * samples_per_frame

        t = np.arange(total_samples) / self.sr

        # Generate smooth pitch curve
        frame_times = np.arange(n_frames) * hop_length / self.sr
        pitch_curve = np.interp(t, frame_times, target_contour)

        # Moderate smoothing
        smooth_window = int(self.sr * 0.03)  # 30ms
        if smooth_window > 0 and len(pitch_curve) > smooth_window:
            kernel = np.hanning(smooth_window)
            kernel /= kernel.sum()
            pitch_curve = np.convolve(pitch_curve, kernel, mode='same')

        # Generate amplitude curve
        amp_curve = np.interp(t, frame_times, rms)

        # Smooth amplitude
        amp_window = int(self.sr * 0.05)  # 50ms
        if amp_window > 0 and len(amp_curve) > amp_window:
            kernel = np.hanning(amp_window)
            kernel /= kernel.sum()
            amp_curve = np.convolve(amp_curve, kernel, mode='same')

        # Subtle breathing
        breath_rate = 0.2  # Hz
        breath = np.sin(2 * np.pi * breath_rate * t) * 0.03 + 1.0
        amp_curve *= breath

        # Continuous phase
        phase = 2 * np.pi * np.cumsum(pitch_curve) / self.sr

        # Rich harmonic series for clear humming character
        y_hum = np.zeros(total_samples)

        # Strong fundamental
        y_hum += np.sin(phase) * 1.4

        # Rich harmonics for human quality
        y_hum += np.sin(2 * phase) * 0.7
        y_hum += np.sin(3 * phase) * 0.35
        y_hum += np.sin(4 * phase) * 0.2
        y_hum += np.sin(5 * phase) * 0.12
        y_hum += np.sin(6 * phase) * 0.07

        # Apply amplitude
        y_hum *= amp_curve

        # Gentle vibrato
        vibrato_rate = 5.5  # Hz
        vibrato_depth = 0.8  # Hz
        vibrato = np.sin(2 * np.pi * vibrato_rate * t) * vibrato_depth
        vibrato_phase = 2 * np.pi * np.cumsum(vibrato) / self.sr
        y_hum *= (1 + 0.025 * np.sin(vibrato_phase))

        # Normalize
        y_hum = y_hum / (np.max(np.abs(y_hum)) + 1e-8)

        return y_hum

    def add_vocal_character(self, y):
        """Add vocal formants and resonances"""
        print("Adding vocal character...")

        # Nasal resonance for closed-mouth humming
        sos_n1 = signal.butter(3, [230, 320], 'bandpass', fs=self.sr, output='sos')
        nasal = signal.sosfilt(sos_n1, y) * 0.4

        # Mid formant for presence
        sos_f = signal.butter(2, [700, 1100], 'bandpass', fs=self.sr, output='sos')
        formant = signal.sosfilt(sos_f, y) * 0.2

        # Upper nasal for character
        sos_n2 = signal.butter(2, [1900, 2400], 'bandpass', fs=self.sr, output='sos')
        upper = signal.sosfilt(sos_n2, y) * 0.12

        # Body resonance
        sos_body = signal.butter(4, [100, 170], 'bandpass', fs=self.sr, output='sos')
        body = signal.sosfilt(sos_body, y) * 0.25

        # Mix
        y_vocal = y + nasal + formant + upper + body

        # Normalize
        y_vocal = y_vocal / (np.max(np.abs(y_vocal)) + 1e-8)

        return y_vocal

    def add_subtle_masking(self, y):
        """Add very subtle background masking - barely noticeable"""
        print("Adding subtle background texture...")

        # Generate very subtle pink noise
        white = np.random.randn(len(y))
        fft = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(len(y), 1.0/self.sr)
        freqs[0] = 1
        pink_fft = fft / np.sqrt(freqs)
        pink = np.fft.irfft(pink_fft, n=len(y))
        pink = pink / (np.max(np.abs(pink)) + 1e-8)

        # Filter to low frequencies only (very subtle rumble)
        sos_low = signal.butter(4, [60, 200], 'bandpass', fs=self.sr, output='sos')
        subtle_rumble = signal.sosfilt(sos_low, pink) * 0.05

        # Modulate by humming envelope
        envelope = np.abs(signal.hilbert(y))
        smooth_env = signal.savgol_filter(envelope, window_length=501, polyorder=3)
        subtle_rumble *= smooth_env

        # Mix with humming - 95% humming, 5% texture
        y_textured = y * 0.95 + subtle_rumble * 0.05

        # Normalize
        y_textured = y_textured / (np.max(np.abs(y_textured)) + 1e-8)

        return y_textured

    def add_binaural_theta(self, y):
        """Add binaural theta waves for relaxation"""
        print("Adding binaural theta effect...")

        # 6 Hz theta for deep relaxation
        beat_freq = 6.0  # Hz
        t = np.arange(len(y)) / self.sr

        # Left channel - original
        left = y.copy()

        # Right channel - subtle modulation
        phase_mod = np.sin(2 * np.pi * beat_freq * t) * 0.015
        right = y * (1 + phase_mod)

        # Stereo
        stereo = np.column_stack([left, right])

        return stereo

    def apply_final_processing(self, y):
        """Final processing for sleep"""
        print("Applying final processing...")

        # Gentle low-pass to smooth
        sos_lp = signal.butter(6, 2800, 'lowpass', fs=self.sr, output='sos')
        y = signal.sosfilt(sos_lp, y)

        # High-pass to clean
        sos_hp = signal.butter(4, 50, 'highpass', fs=self.sr, output='sos')
        y = signal.sosfilt(sos_hp, y)

        # Smooth envelope
        analytic = signal.hilbert(y)
        envelope = np.abs(analytic)

        window_samples = int(self.sr * 0.04)  # 40ms
        if window_samples > 0:
            kernel = np.hanning(window_samples)
            kernel /= kernel.sum()
            smooth_env = np.convolve(envelope, kernel, mode='same')
        else:
            smooth_env = envelope

        phase = np.angle(analytic)
        y = smooth_env * np.cos(phase)

        # Gentle compression for consistency
        threshold = 0.4
        ratio = 2.5
        mask = np.abs(y) > threshold
        y[mask] = np.sign(y[mask]) * (threshold + (np.abs(y[mask]) - threshold) / ratio)

        # Gentle warmth
        y = np.tanh(y * 1.25) / np.tanh(1.25)

        # De-click
        from scipy.ndimage import median_filter
        y = median_filter(y, size=3)

        # Final polish
        sos_final = signal.butter(4, 3500, 'lowpass', fs=self.sr, output='sos')
        y = signal.sosfilt(sos_final, y)

        # Normalize to -3dB for comfortable volume
        y = y / (np.max(np.abs(y)) + 1e-8) * 0.7

        return y

    def convert(self, input_file, output_file, add_binaural=True, verbose=True):
        """Main conversion function"""

        if verbose:
            print("="*70)
            print("BALANCED SLEEP HUMMING CONVERTER")
            print("="*70)
            print(f"\nLoading: {input_file}")

        # Load audio
        y, sr = librosa.load(input_file, sr=self.sr, mono=True)

        if verbose:
            print(f"Duration: {len(y) / sr:.2f}s | Sample Rate: {sr} Hz\n")

        # Extract pitch contour
        f0_contour, rms, voiced_flag, hop_length = self.extract_pitch_contour(y)

        # Synthesize humming
        y_hum = self.synthesize_balanced_humming(f0_contour, rms, voiced_flag, hop_length)

        # Add vocal character
        y_hum = self.add_vocal_character(y_hum)

        # Add very subtle background texture
        y_hum = self.add_subtle_masking(y_hum)

        # Process for sleep
        y_final = self.apply_final_processing(y_hum)

        # Add binaural if requested
        if add_binaural:
            y_final = self.add_binaural_theta(y_final)

        # Save
        sf.write(output_file, y_final, sr)

        if verbose:
            print(f"\nSaved: {output_file}")
            print("="*70)
            print("\nBALANCED SLEEP HUMMING FEATURES:")
            print(f"  - Target frequency: {self.target_freq} Hz")
            print("  - Clear humming character with 6 harmonics")
            print("  - 30% pitch expression (natural, not robotic)")
            print("  - Vocal formants for human warmth")
            print("  - Very subtle background texture (5% only)")
            print("  - 6 Hz binaural theta waves" if add_binaural else "  - Mono output")
            print("  - Smooth but not drone-like")
            print("  - Perfect for meditation and sleep")
            print("="*70)


def main():
    import sys

    if len(sys.argv) < 3:
        print("Usage: python balanced_sleep_humming.py <input> <output> [--no-binaural]")
        print("Example: python balanced_sleep_humming.py vocal.wav sleep.wav")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    add_binaural = '--no-binaural' not in sys.argv

    converter = BalancedSleepHummingConverter(target_freq=130, sr=44100)
    converter.convert(input_file, output_file, add_binaural=add_binaural, verbose=True)


if __name__ == "__main__":
    main()
