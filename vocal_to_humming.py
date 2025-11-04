#!/usr/bin/env python3
"""
Automated Vocal to Humming Converter
Iteratively processes vocal audio until it meets humming criteria
"""

import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from scipy.ndimage import uniform_filter1d
import warnings
warnings.filterwarnings('ignore')

class VocalToHummingConverter:
    """Converts vocal with words to pure therapeutic humming at 130 Hz"""

    def __init__(self, target_freq=130, sr=44100):
        self.target_freq = target_freq
        self.sr = sr

        # Target criteria for validation
        self.criteria = {
            'consonant_energy_ratio': 0.15,  # Max 15% energy in consonant range
            'sibilance_ratio': 0.05,          # Max 5% sibilance
            'transient_ratio': 0.10,          # Max 10% transient content
            'spectral_centroid': 500,         # Target low centroid (Hz)
            'fundamental_prominence': 0.30,   # Min 30% energy at fundamental
        }

    def analyze_audio(self, y):
        """Analyze audio and return metrics"""
        # Compute STFT
        D = librosa.stft(y, n_fft=2048, hop_length=512)
        magnitude = np.abs(D)

        # Frequency bins
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=2048)

        # Define frequency ranges
        consonant_range = (2000, 4000)  # Hz
        sibilance_range = (4000, 8000)  # Hz
        fundamental_range = (self.target_freq - 20, self.target_freq + 20)

        # Calculate energy in each range
        def get_energy_in_range(freq_range):
            mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
            return np.sum(magnitude[mask, :])

        total_energy = np.sum(magnitude)
        consonant_energy = get_energy_in_range(consonant_range)
        sibilance_energy = get_energy_in_range(sibilance_range)
        fundamental_energy = get_energy_in_range(fundamental_range)

        # Spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=self.sr)[0]
        avg_centroid = np.mean(spectral_centroids)

        # Transient detection (onset strength)
        onset_env = librosa.onset.onset_strength(y=y, sr=self.sr)
        transient_ratio = np.sum(onset_env > np.percentile(onset_env, 75)) / len(onset_env)

        metrics = {
            'consonant_energy_ratio': consonant_energy / total_energy,
            'sibilance_ratio': sibilance_energy / total_energy,
            'transient_ratio': transient_ratio,
            'spectral_centroid': avg_centroid,
            'fundamental_prominence': fundamental_energy / total_energy,
        }

        return metrics

    def calculate_score(self, metrics):
        """Calculate how well the audio meets criteria (0-1, higher is better)"""
        scores = []

        # Consonant energy (lower is better)
        scores.append(max(0, 1 - (metrics['consonant_energy_ratio'] / self.criteria['consonant_energy_ratio'])))

        # Sibilance (lower is better)
        scores.append(max(0, 1 - (metrics['sibilance_ratio'] / self.criteria['sibilance_ratio'])))

        # Transient ratio (lower is better)
        scores.append(max(0, 1 - (metrics['transient_ratio'] / self.criteria['transient_ratio'])))

        # Spectral centroid (closer to target is better)
        centroid_diff = abs(metrics['spectral_centroid'] - self.criteria['spectral_centroid'])
        scores.append(max(0, 1 - (centroid_diff / 1000)))

        # Fundamental prominence (higher is better)
        scores.append(min(1, metrics['fundamental_prominence'] / self.criteria['fundamental_prominence']))

        return np.mean(scores)

    def apply_processing(self, y, params):
        """Apply processing with given parameters"""
        y_processed = y.copy()

        # 1. Pitch shift to target frequency
        # Estimate original pitch
        pitches, magnitudes = librosa.piptrack(y=y, sr=self.sr)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)

        if pitch_values:
            avg_pitch = np.median(pitch_values)
            n_steps = 12 * np.log2(self.target_freq / avg_pitch)
            y_processed = librosa.effects.pitch_shift(y_processed, sr=self.sr, n_steps=n_steps)

        # 2. Formant shifting (simulate by time-stretching without pitch change)
        formant_shift = params['formant_shift']
        if formant_shift != 1.0:
            y_processed = librosa.effects.time_stretch(y_processed, rate=formant_shift)
            # Resample back to original length
            target_length = len(y)
            if len(y_processed) != target_length:
                y_processed = librosa.resample(y_processed,
                                               orig_sr=len(y_processed),
                                               target_sr=target_length)

        # 3. Aggressive low-pass filter
        cutoff_freq = params['lowpass_cutoff']
        sos = signal.butter(params['lowpass_order'], cutoff_freq, 'low', fs=self.sr, output='sos')
        y_processed = signal.sosfilt(sos, y_processed)

        # 4. High-pass filter
        sos_hp = signal.butter(4, 40, 'high', fs=self.sr, output='sos')
        y_processed = signal.sosfilt(sos_hp, y_processed)

        # 5. Notch filters for consonant frequencies
        for notch_freq in params['notch_frequencies']:
            Q = params['notch_q']
            b_notch, a_notch = signal.iirnotch(notch_freq, Q, self.sr)
            y_processed = signal.filtfilt(b_notch, a_notch, y_processed)

        # 6. Transient suppression (envelope smoothing)
        if params['transient_suppression'] > 0:
            # Get envelope
            analytic_signal = signal.hilbert(y_processed)
            envelope = np.abs(analytic_signal)

            # Smooth envelope
            window_size = int(self.sr * params['transient_suppression'] / 1000)  # ms to samples
            smoothed_envelope = uniform_filter1d(envelope, size=window_size)

            # Apply smoothed envelope
            phase = np.angle(analytic_signal)
            y_processed = smoothed_envelope * np.cos(phase)

        # 7. Compression (dynamic range reduction)
        threshold = params['compression_threshold']
        ratio = params['compression_ratio']

        # Simple compression
        mask = np.abs(y_processed) > threshold
        y_processed[mask] = np.sign(y_processed[mask]) * (
            threshold + (np.abs(y_processed[mask]) - threshold) / ratio
        )

        # 8. Emphasize fundamental frequency
        # Bandpass filter around 130 Hz
        sos_bp = signal.butter(2, [self.target_freq - 30, self.target_freq + 30],
                               'bandpass', fs=self.sr, output='sos')
        fundamental = signal.sosfilt(sos_bp, y_processed)

        # Mix back with boost
        y_processed = y_processed + fundamental * params['fundamental_boost']

        # Normalize
        y_processed = y_processed / (np.max(np.abs(y_processed)) + 1e-8)

        return y_processed

    def optimize_parameters(self, y, max_iterations=50, verbose=True):
        """Iteratively optimize processing parameters"""

        # Initial parameter set
        best_params = {
            'formant_shift': 0.75,        # Lower formants
            'lowpass_cutoff': 2500,       # Hz
            'lowpass_order': 6,           # Filter steepness
            'notch_frequencies': [2500, 4000, 6000],  # Consonant frequencies
            'notch_q': 5,                 # Notch filter Q factor
            'transient_suppression': 20,  # ms
            'compression_threshold': 0.3,
            'compression_ratio': 4.0,
            'fundamental_boost': 0.3,
        }

        best_score = 0
        best_audio = None

        # Parameter search space
        param_ranges = {
            'formant_shift': [0.65, 0.70, 0.75, 0.80, 0.85],
            'lowpass_cutoff': [2000, 2500, 3000, 3500],
            'lowpass_order': [4, 6, 8],
            'notch_q': [3, 5, 8, 10],
            'transient_suppression': [10, 20, 30, 40],
            'compression_ratio': [3.0, 4.0, 5.0, 6.0],
            'fundamental_boost': [0.2, 0.3, 0.4, 0.5],
        }

        iteration = 0

        if verbose:
            print("Starting optimization...")
            print("-" * 60)

        # Initial evaluation
        y_test = self.apply_processing(y, best_params)
        metrics = self.analyze_audio(y_test)
        best_score = self.calculate_score(metrics)
        best_audio = y_test

        if verbose:
            print(f"Initial Score: {best_score:.3f}")
            print(f"Metrics: {metrics}")
            print()

        # Iterative optimization (greedy search)
        for iteration in range(max_iterations):
            improved = False

            # Try varying each parameter
            for param_name, param_values in param_ranges.items():
                for value in param_values:
                    # Skip if same as current
                    if best_params[param_name] == value:
                        continue

                    # Test new parameters
                    test_params = best_params.copy()
                    test_params[param_name] = value

                    try:
                        y_test = self.apply_processing(y, test_params)
                        metrics = self.analyze_audio(y_test)
                        score = self.calculate_score(metrics)

                        # If better, update
                        if score > best_score:
                            best_score = score
                            best_params = test_params
                            best_audio = y_test
                            improved = True

                            if verbose:
                                print(f"Iteration {iteration + 1}: Improved! Score: {best_score:.3f}")
                                print(f"  Changed {param_name} to {value}")
                                print(f"  Metrics: {metrics}")
                                print()
                    except Exception as e:
                        if verbose:
                            print(f"  Error with {param_name}={value}: {e}")
                        continue

            # Check if we've met criteria
            if best_score > 0.85:  # 85% of optimal
                if verbose:
                    print(f"Target criteria met! Final score: {best_score:.3f}")
                break

            # If no improvement, we've converged
            if not improved:
                if verbose:
                    print(f"Converged at iteration {iteration + 1}. Final score: {best_score:.3f}")
                break

        if verbose:
            print("-" * 60)
            print("Optimization complete!")
            print(f"Final parameters: {best_params}")
            print(f"Final score: {best_score:.3f}")

        return best_audio, best_params, best_score

    def convert(self, input_file, output_file, max_iterations=50, verbose=True):
        """Main conversion function"""

        if verbose:
            print(f"Loading audio from: {input_file}")

        # Load audio
        y, sr = librosa.load(input_file, sr=self.sr, mono=True)

        if verbose:
            print(f"Audio duration: {len(y) / sr:.2f} seconds")
            print(f"Sample rate: {sr} Hz")
            print()

        # Analyze original
        if verbose:
            print("Analyzing original audio...")
            original_metrics = self.analyze_audio(y)
            original_score = self.calculate_score(original_metrics)
            print(f"Original score: {original_score:.3f}")
            print(f"Original metrics: {original_metrics}")
            print()

        # Optimize
        y_processed, best_params, final_score = self.optimize_parameters(
            y, max_iterations=max_iterations, verbose=verbose
        )

        # Save result
        if verbose:
            print(f"Saving processed audio to: {output_file}")

        sf.write(output_file, y_processed, sr)

        if verbose:
            print("Done!")
            print()
            print("=" * 60)
            print("RESULTS:")
            print(f"  Original Score: {original_score:.3f}")
            print(f"  Final Score: {final_score:.3f}")
            print(f"  Improvement: {((final_score - original_score) / original_score * 100):.1f}%")
            print("=" * 60)

        return best_params, final_score


def main():
    """Example usage"""
    import sys

    if len(sys.argv) < 3:
        print("Usage: python vocal_to_humming.py <input_file> <output_file> [max_iterations]")
        print("Example: python vocal_to_humming.py vocal.wav humming.wav 30")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    max_iterations = int(sys.argv[3]) if len(sys.argv) > 3 else 50

    converter = VocalToHummingConverter(target_freq=130, sr=44100)
    converter.convert(input_file, output_file, max_iterations=max_iterations, verbose=True)


if __name__ == "__main__":
    main()
