from nuspacesim import eas_optical

import numpy as np


def test_known_values():
    altitude = np.arange(0, 21, dtype=np.float32)
    beta_tr = np.full_like(altitude, np.radians(42.0), dtype=np.float32)
    cpa = eas_optical.eas.CphotAng()

    dphots, cang = cpa(beta_tr, altitude)

    # print(dphots)
    # print(cang)

    old_dphots = np.array(
        [
            1.6210271e03,
            1.8776989e03,
            2.1580850e03,
            2.4297334e03,
            2.6037090e03,
            2.5893950e03,
            2.6665376e03,
            1.9930570e03,
            1.2905229e03,
            7.2669135e02,
            3.5919400e02,
            1.5781609e02,
            6.3526283e01,
            2.4607666e01,
            9.4170065e00,
            3.5231948e00,
            1.3178004e00,
            5.0189447e-01,
            1.9809218e-01,
            8.2580306e-02,
            3.6842916e-02,
        ]
    )

    old_cang = np.array(
        [
            1.1297,
            1.0476834,
            0.9589647,
            0.86524945,
            0.7722871,
            0.68627226,
            0.6104225,
            0.54487425,
            0.48861432,
            0.4402858,
            0.39882657,
            0.3633839,
            0.33331406,
            0.30719605,
            0.28355855,
            0.2634559,
            0.24582747,
            0.22999273,
            0.21532202,
            0.20063952,
            0.18477382,
        ]
    )

    assert np.allclose(dphots, old_dphots)
    assert np.allclose(cang, old_cang)


if __name__ == "__main__":
    test_known_values()
