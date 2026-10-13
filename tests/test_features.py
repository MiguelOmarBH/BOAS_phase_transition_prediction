import numpy as np

from boas_transition.features import eeg_features, pulse_features


def test_frequency_content_and_microvolt_scale():
    sampling_rate = 256
    time = np.arange(30 * sampling_rate) / sampling_rate
    alpha = 30 * np.sin(2 * np.pi * 10 * time)
    theta = 30 * np.sin(2 * np.pi * 6 * time)
    signals = np.array([[alpha, alpha], [theta, theta]])
    result = eeg_features(signals, sampling_rate)
    assert result["eeg_log_alpha_theta_mean"][0] > result["eeg_log_alpha_theta_mean"][1]
    expected_log_rms = np.log10(30 / np.sqrt(2))
    np.testing.assert_allclose(result["eeg_log_rms_mean"], expected_log_rms, atol=0.01)
    assert all(np.isfinite(values).all() for values in result.values())


def test_regular_pulse_rate_and_invalid_signal():
    sampling_rate = 256
    time = np.arange(30 * sampling_rate) / sampling_rate
    result = pulse_features(np.sin(2 * np.pi * 1.2 * time), sampling_rate)
    assert abs(result["ppg_hr"] - 72) < 1
    assert result["ppg_rmssd"] < 10
    invalid = pulse_features(np.zeros_like(time), sampling_rate)
    assert np.isnan(invalid["ppg_hr"])
