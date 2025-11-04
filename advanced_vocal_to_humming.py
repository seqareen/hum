#!/usr/bin/env python3
"""
Advanced Vocal to Humming Converter
Uses spectral analysis and adaptive filtering for superior consonant removal
"""

import numpy as np
import librosa
import soundfile as sf
from scipy import signal, ndimage
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')


class AdvancedVocalToHummingConverter:
    """Advanced converter with spectral processing and adaptive filtering"""

    def __init__(self, target_freq=130, sr=44100):
        self.target_freq = target_freq
        self.sr = sr

        # More stringent criteria
        self.criteria = {
            'consonant_energy_ratio': 0.10,   # Max 10%
            'sibilance_ratio': 0.03,           # Max 3%
            'transient_density': 0.05,         # Max 5% high transients
            'spectral_centroid': 400,          # Lower target
            'spectral_flatness': 0.1,          # More tonal (less noisy)
            'fundamental_prominence': 0.35,    # Min 35%
            'high_freq_rolloff': -40,          # dB at 4kHz relative to fundamental
        }

    def spectral_analysis(self, y):
        """Detailed spectral analysis"""
        # STFT with higher resolution
        D = librosa.stft(y, n_fft=4096, hop_length=512)
        magnitude = np.abs(D)
        phase = np.angle(D)
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=4096)

        # Frequency ranges
        ranges = {
            'fundamental': (self.target_freq - 30, self.target_freq + 30),
            'harmonics': (self.target_freq * 2 - 50, self.target_freq * 4 + 50),
            'consonants': (1800, 4500),
            'sibilance': (4500, 9000),
            'ultra_high': (9000, 20000),
        }

        # Calculate energy in each range
        energies = {}
        total_energy = np.sum(magnitude ** 2)

        for name, (f_low, f_high) in ranges.items():
            mask = (freqs >= f_low) & (freqs <= f_high)
            energies[name] = np.sum(magnitude[mask, :] ** 2)

        # Metrics
        consonant_ratio = energies['consonants'] / total_energy
        sibilance_ratio = energies['sibilance'] / total_energy
        fundamental_prominence = energies['fundamental'] / total_energy

        # Spectral centroid (weighted mean frequency)
        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=self.sr)[0]
        avg_centroid = np.mean(spec_centroid)

        # Spectral flatness (tonality vs noisiness)
        spec_flatness = librosa.feature.spectral_flatness(y=y)[0]
        avg_flatness = np.mean(spec_flatness)

        # Transient analysis
        onset_env = librosa.onset.onset_strength(y=y, sr=self.sr, aggregate=np.median)
        strong_transients = np.sum(onset_env > np.percentile(onset_env, 80))
        transient_density = strong_transients / len(onset_env)

        # High frequency rolloff
        fund_energy = energies['fundamental'] / (np.sum(magnitude[:, 0] ** 2) + 1e-10)
        high_freq_mask = freqs > 4000
        high_energy = np.sum(magnitude[high_freq_mask, :] ** 2)
        high_ratio = high_energy / (fund_energy + 1e-10)
        high_freq_rolloff = 10 * np.log10(high_ratio + 1e-10)

        metrics = {
            'consonant_energy_ratio': consonant_ratio,
            'sibilance_ratio': sibilance_ratio,
            'transient_density': transient_density,
            'spectral_centroid': avg_centroid,
            'spectral_flatness': avg_flatness,
            'fundamental_prominence': fundamental_prominence,
            'high_freq_rolloff': high_freq_rolloff,
        }

        return metrics, D, magnitude, phase, freqs

    def detect_consonant_regions(self, y):
        """Detect time regions with strong consonants"""
        # Compute spectral flux (rate of spectral change)
        S = np.abs(librosa.stft(y))
        flux = np.sqrt(np.sum(np.diff(S, axis=1) ** 2, axis=0))
        flux = np.concatenate(([0], flux))

        # High frequency energy per frame
        freqs = librosa.fft_frequencies(sr=self.sr)
        high_freq_mask = freqs > 2000
        high_energy = np.sum(S[high_freq_mask, :], axis=0)

        # Normalize
        flux_norm = (flux - np.min(flux)) / (np.max(flux) - np.min(flux) + 1e-10)
        high_norm = (high_energy - np.min(high_energy)) / (np.max(high_energy) - np.min(high_energy) + 1e-10)

        # Combined score
        consonant_score = 0.5 * flux_norm + 0.5 * high_norm

        # Threshold
        threshold = np.percentile(consonant_score, 70)
        consonant_frames = consonant_score > threshold

        return consonant_frames, consonant_score

    def spectral_gate(self, D, magnitude, phase, consonant_frames):
        """Apply spectral gating to reduce consonants"""
        D_gated = D.copy()

        # For consonant frames, reduce high frequencies
        for i, is_consonant in enumerate(consonant_frames):
            if is_consonant and i < D_gated.shape[1]:
                # Create frequency-dependent attenuation
                freqs = librosa.fft_frequencies(sr=self.sr, n_fft=D.shape[0] * 2 - 2)
                attenuation = np.ones(len(freqs))

                # Reduce high frequencies more aggressively
                attenuation[freqs > 2000] *= 0.3
                attenuation[freqs > 4000] *= 0.1
                attenuation[freqs > 6000] *= 0.05

                D_gated[:, i] *= attenuation[:D_gated.shape[0]]

        return D_gated

    def adaptive_processing(self, y, params):
        """Apply adaptive processing with spectral manipulation"""

        # 1. Detect consonants
        consonant_frames, consonant_score = self.detect_consonant_regions(y)

        # 2. Spectral processing
        metrics, D, magnitude, phase, freqs = self.spectral_analysis(y)

        # 3. Apply spectral gate
        D_gated = self.spectral_gate(D, magnitude, phase, consonant_frames)

        # 4. Reconstruct from modified spectrogram
        y_gated = librosa.istft(D_gated)

        # 5. Pitch shift to target
        pitches, mags = librosa.piptrack(y=y_gated, sr=self.sr, fmin=50, fmax=500)
        pitch_values = []
        for t in range(min(100, pitches.shape[1])):  # Sample first 100 frames
            index = mags[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 50:
                pitch_values.append(pitch)

        if pitch_values:
            avg_pitch = np.median(pitch_values)
            n_steps = 12 * np.log2(self.target_freq / max(avg_pitch, 50))
            n_steps = np.clip(n_steps, -36, 12)  # Reasonable range
            y_gated = librosa.effects.pitch_shift(y_gated, sr=self.sr, n_steps=n_steps)

        # 6. Formant shifting via time stretch
        if params['formant_shift'] != 1.0:
            y_gated = librosa.effects.time_stretch(y_gated, rate=params['formant_shift'])
            if len(y_gated) != len(y):
                # Resample to original length
                y_gated = signal.resample(y_gated, len(y))

        # 7. Multi-stage filtering
        # High-pass
        sos_hp = signal.butter(4, 40, 'highpass', fs=self.sr, output='sos')
        y_filtered = signal.sosfilt(sos_hp, y_gated)

        # Low-pass (very aggressive)
        cutoff = params['lowpass_cutoff']
        order = params['lowpass_order']
        sos_lp = signal.butter(order, cutoff, 'lowpass', fs=self.sr, output='sos')
        y_filtered = signal.sosfilt(sos_lp, y_filtered)

        # Multiple notch filters for problem frequencies
        for freq in params['notch_frequencies']:
            Q = params['notch_q']
            b, a = signal.iirnotch(freq, Q, self.sr)
            y_filtered = signal.filtfilt(b, a, y_filtered)

        # 8. Transient suppression via envelope smoothing
        if params['smooth_envelope_ms'] > 0:
            # Extract envelope
            analytic = signal.hilbert(y_filtered)
            envelope = np.abs(analytic)

            # Smooth with moving average
            window_samples = int(self.sr * params['smooth_envelope_ms'] / 1000)
            kernel = np.ones(window_samples) / window_samples
            smooth_env = np.convolve(envelope, kernel, mode='same')

            # Apply
            phase_angle = np.angle(analytic)
            y_filtered = smooth_env * np.cos(phase_angle)

        # 9. Emphasize fundamental
        # Bandpass around target frequency
        bw = params['fundamental_bandwidth']
        low = max(20, self.target_freq - bw)
        high = min(self.sr // 2 - 100, self.target_freq + bw)
        sos_bp = signal.butter(4, [low, high], 'bandpass', fs=self.sr, output='sos')
        fundamental = signal.sosfilt(sos_bp, y_filtered)

        # Mix back
        boost = params['fundamental_boost']
        y_enhanced = y_filtered + fundamental * boost

        # 10. Soft compression
        threshold = params['compression_threshold']
        ratio = params['compression_ratio']

        # Soft knee compression
        def compress(x, thresh, ratio):
            mask = np.abs(x) > thresh
            compressed = x.copy()
            compressed[mask] = np.sign(x[mask]) * (
                thresh + (np.abs(x[mask]) - thresh) / ratio
            )
            return compressed

        y_compressed = compress(y_enhanced, threshold, ratio)

        # 11. Final smoothing (remove any remaining clicks)
        y_smoothed = signal.savgol_filter(y_compressed,
                                          window_length=min(51, len(y_compressed) // 10 * 2 + 1),
                                          polyorder=3)

        # Normalize
        y_final = y_smoothed / (np.max(np.abs(y_smoothed)) + 1e-8) * 0.95

        return y_final

    def calculate_score(self, metrics):
        """Calculate quality score (0-1)"""
        scores = []
        weights = []

        # Each criterion with weight
        criteria_weights = {
            'consonant_energy_ratio': 0.25,
            'sibilance_ratio': 0.20,
            'transient_density': 0.15,
            'spectral_centroid': 0.15,
            'spectral_flatness': 0.10,
            'fundamental_prominence': 0.10,
            'high_freq_rolloff': 0.05,
        }

        for key, weight in criteria_weights.items():
            target = self.criteria[key]
            actual = metrics[key]

            if key in ['consonant_energy_ratio', 'sibilance_ratio', 'transient_density',
                       'spectral_flatness', 'high_freq_rolloff']:
                # Lower is better
                if key == 'high_freq_rolloff':
                    # More negative is better
                    score = max(0, min(1, 1 - (actual - target) / 20))
                else:
                    score = max(0, 1 - actual / target)
            elif key == 'spectral_centroid':
                # Closer to target is better
                diff = abs(actual - target)
                score = max(0, 1 - diff / 1000)
            else:  # fundamental_prominence
                # Higher is better
                score = min(1, actual / target)

            scores.append(score * weight)
            weights.append(weight)

        return sum(scores)

    def optimize(self, y, max_iterations=30, verbose=True):
        """Optimize processing parameters"""

        # Parameter ranges for search
        param_space = {
            'formant_shift': [0.65, 0.70, 0.75, 0.80],
            'lowpass_cutoff': [1800, 2200, 2500, 2800, 3000],
            'lowpass_order': [6, 8, 10],
            'notch_frequencies': [
                [2500, 4000, 6500],
                [2000, 3500, 5500],
                [2200, 4200, 6800],
            ],
            'notch_q': [5, 8, 12],
            'smooth_envelope_ms': [15, 25, 35, 45],
            'fundamental_bandwidth': [25, 35, 45],
            'fundamental_boost': [0.3, 0.5, 0.7],
            'compression_threshold': [0.25, 0.35, 0.45],
            'compression_ratio': [3.5, 4.5, 5.5],
        }

        # Start with middle values
        best_params = {k: v[len(v) // 2] for k, v in param_space.items()}

        # Initial evaluation
        y_test = self.adaptive_processing(y, best_params)
        metrics, _, _, _, _ = self.spectral_analysis(y_test)
        best_score = self.calculate_score(metrics)
        best_audio = y_test

        if verbose:
            print("=" * 70)
            print("ADVANCED OPTIMIZATION")
            print("=" * 70)
            print(f"Initial Score: {best_score:.4f}")
            print(f"Target: >0.850")
            print()

        for iteration in range(max_iterations):
            improved = False

            for param_name, values in param_space.items():
                for value in values:
                    if best_params[param_name] == value:
                        continue

                    test_params = best_params.copy()
                    test_params[param_name] = value

                    try:
                        y_test = self.adaptive_processing(y, test_params)
                        metrics, _, _, _, _ = self.spectral_analysis(y_test)
                        score = self.calculate_score(metrics)

                        if score > best_score:
                            best_score = score
                            best_params = test_params
                            best_audio = y_test
                            improved = True

                            if verbose:
                                print(f"[Iteration {iteration + 1:2d}] Score: {score:.4f} (+{score - (score - (score - best_score)):.4f}) | {param_name}={value}")

                    except Exception as e:
                        continue

            # Check convergence
            if best_score > 0.85:
                if verbose:
                    print()
                    print("✓ Target criteria achieved!")
                break

            if not improved:
                if verbose:
                    print()
                    print(f"Converged at score: {best_score:.4f}")
                break

        # Final metrics
        final_metrics, _, _, _, _ = self.spectral_analysis(best_audio)

        if verbose:
            print()
            print("=" * 70)
            print("FINAL RESULTS")
            print("=" * 70)
            print(f"Final Score: {best_score:.4f}")
            print()
            print("Metrics vs Targets:")
            for key in self.criteria:
                target = self.criteria[key]
                actual = final_metrics[key]
                status = "✓" if (actual <= target if key != 'fundamental_prominence' else actual >= target) else "✗"
                print(f"  {status} {key:25s}: {actual:8.3f} (target: {target:.3f})")
            print("=" * 70)

        return best_audio, best_params, best_score, final_metrics

    def convert(self, input_file, output_file, max_iterations=30, verbose=True):
        """Main conversion function"""

        if verbose:
            print(f"\nLoading: {input_file}")

        y, sr = librosa.load(input_file, sr=self.sr, mono=True)

        if verbose:
            print(f"Duration: {len(y) / sr:.2f}s | Sample Rate: {sr} Hz")

        # Process
        y_final, params, score, metrics = self.optimize(y, max_iterations, verbose)

        # Save
        sf.write(output_file, y_final, sr)

        if verbose:
            print(f"\nSaved: {output_file}")

        return params, score, metrics


def main():
    import sys

    if len(sys.argv) < 3:
        print("Usage: python advanced_vocal_to_humming.py <input> <output> [iterations]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    max_iter = int(sys.argv[3]) if len(sys.argv) > 3 else 30

    converter = AdvancedVocalToHummingConverter(target_freq=130, sr=44100)
    converter.convert(input_file, output_file, max_iterations=max_iter, verbose=True)


if __name__ == "__main__":
    main()
