import numpy as np
import pandas as pd

from dataload.scaler import StandardScaler


FEATURES = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]


def _etth1_values(etth1_path):
    return pd.read_csv(etth1_path, usecols=FEATURES).to_numpy(dtype=np.float32)


def test_scaler_fits_only_supplied_train_rows(etth1_path):
    values = _etth1_values(etth1_path)
    train_end = int(len(values) * 0.7)
    scaler = StandardScaler().fit(values[:train_end])

    transformed_train = scaler.transform(values[:train_end])
    transformed_all = scaler.transform(values)

    np.testing.assert_allclose(transformed_train.mean(axis=0), 0.0, atol=2e-6)
    assert not np.allclose(transformed_all.mean(axis=0), 0.0, atol=1e-3)


def test_scaler_state_and_m_inverse_transform_round_trip(etth1_path):
    values = _etth1_values(etth1_path)
    scaler = StandardScaler().fit(values[:1000])
    restored_scaler = StandardScaler()
    restored_scaler.load_state_dict(scaler.state_dict())

    sample = values[1000:1012].reshape(2, 6, 7)
    restored = restored_scaler.inverse_transform(restored_scaler.transform(sample))

    assert restored.shape == sample.shape
    np.testing.assert_allclose(restored, sample, rtol=1e-5, atol=1e-5)


def test_ms_interval_inverse_transform_uses_only_last_feature(etth1_path):
    values = _etth1_values(etth1_path)
    scaler = StandardScaler().fit(values[:1000])
    scaled = np.zeros((2, 4, 1, 3), dtype=np.float32)

    restored = scaler.inverse_transform(scaled, channel_indices=[-1])

    assert restored.shape == (2, 4, 1, 3)
    np.testing.assert_allclose(restored, scaler.mean_[-1])


def test_zero_variance_channel_stays_finite():
    values = np.array([[1.0, 2.0], [1.0, 4.0], [1.0, 6.0]])
    scaler = StandardScaler(epsilon=1.0e-8).fit(values)
    transformed = scaler.transform(values)

    assert np.isfinite(transformed).all()
    np.testing.assert_allclose(transformed[:, 0], 0.0)
