"""WADA-SNR estimation for speech audio quality analysis."""

import numpy as np


def wada_snr(audio: np.ndarray) -> float:
    """
    Estimate speech SNR in dB using WADA-SNR.

    Args:
        audio: Mono audio waveform as a NumPy array.

    Returns:
        Estimated SNR in decibels.

    Raises:
        ValueError: If audio is empty, invalid, or silent.
    """

    eps = 1e-10

    # Convert input to a NumPy array
    audio = np.asarray(audio, dtype=np.float64)

    if audio.size == 0:
        raise ValueError("Audio signal is empty.")

    if not np.isfinite(audio).all():
        raise ValueError("Audio contains NaN or Inf values.")

    # Convert stereo to mono if necessary
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if audio.ndim != 1:
        raise ValueError("Audio must be mono or stereo.")

    # Reject silent audio
    peak = np.max(np.abs(audio))

    if peak < eps:
        raise ValueError("Cannot estimate SNR for silent audio.")

    # Normalize waveform for numerical stability
    audio = audio / peak

    abs_audio = np.maximum(np.abs(audio), eps)

    # Calculate waveform amplitude statistics
    v1 = max(abs_audio.mean(), eps)
    v2 = np.log(abs_audio).mean()
    v3 = np.log(v1) - v2

    # WADA-SNR reference lookup table
    db_vals = np.arange(-20, 101)

    g_vals = np.array(
        [
            0.40974774,
            0.40986926,
            0.40998566,
            0.40969089,
            0.40986186,
            0.40999006,
            0.41027138,
            0.41052627,
            0.41101024,
            0.41143264,
            0.41231718,
            0.41337272,
            0.41526426,
            0.4178192,
            0.42077252,
            0.42452799,
            0.42918886,
            0.43510373,
            0.44234195,
            0.45161485,
            0.46221153,
            0.47491647,
            0.48883809,
            0.50509236,
            0.52353709,
            0.54372088,
            0.56532427,
            0.58847532,
            0.61346212,
            0.63954496,
            0.66750818,
            0.69583724,
            0.72454762,
            0.75414799,
            0.78323148,
            0.81240985,
            0.84219775,
            0.87166406,
            0.90030504,
            0.92880418,
            0.95655449,
            0.9835349,
            1.01047155,
            1.0362095,
            1.06136425,
            1.08579312,
            1.1094819,
            1.13277995,
            1.15472826,
            1.17627308,
            1.19703503,
            1.21671694,
            1.23535898,
            1.25364313,
            1.27103891,
            1.28718029,
            1.30302865,
            1.31839527,
            1.33294817,
            1.34700935,
            1.3605727,
            1.37345513,
            1.38577122,
            1.39733504,
            1.40856397,
            1.41959619,
            1.42983624,
            1.43958467,
            1.44902176,
            1.45804831,
            1.46669568,
            1.47486938,
            1.48269965,
            1.49034339,
            1.49748214,
            1.50435106,
            1.51076426,
            1.51698915,
            1.5229097,
            1.528578,
            1.53389835,
            1.5391211,
            1.5439065,
            1.54858517,
            1.55310776,
            1.55744391,
            1.56164927,
            1.56566348,
            1.56938671,
            1.57307767,
            1.57654764,
            1.57980083,
            1.58304129,
            1.58602496,
            1.58880681,
            1.59162477,
            1.5941969,
            1.59693155,
            1.599446,
            1.60185011,
            1.60408668,
            1.60627134,
            1.60826199,
            1.61004547,
            1.61192472,
            1.61369656,
            1.61534074,
            1.61688905,
            1.61838916,
            1.61985374,
            1.62135878,
            1.62268119,
            1.62390423,
            1.62513143,
            1.62632463,
            1.6274027,
            1.62842767,
            1.62945532,
            1.6303307,
            1.63128026,
            1.63204102,
        ]
    )

    # Find the closest reference interval
    indices = np.where(g_vals < v3)[0]

    if len(indices) == 0:
        return float(db_vals[0])

    idx = indices[-1]

    if idx == len(db_vals) - 1:
        return float(db_vals[-1])

    # Linear interpolation
    snr = db_vals[idx] + ((v3 - g_vals[idx]) / (g_vals[idx + 1] - g_vals[idx])) * (
        db_vals[idx + 1] - db_vals[idx]
    )

    return float(snr)
