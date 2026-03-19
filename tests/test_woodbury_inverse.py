from latent_svi_spatial.data.synthetic import (
    SyntheticConfig,
    generate_synthetic_panel,
)
from latent_svi_spatial.models.sar import check_inverse_consistency


def test_inverse_identity() -> None:
    data = generate_synthetic_panel(
        SyntheticConfig(
            n=20,
            r=3,
            seed=123,
            normalize_w="none",
            rho=0.01,
        )
    )

    y = data.y[0]

    err = check_inverse_consistency(
        data.W,
        data.H,
        data.C,
        float(data.rho),
        y,
    )

    assert err < 1e-5