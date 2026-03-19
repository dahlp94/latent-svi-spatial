from latent_svi_spatial.data.synthetic import (
    SyntheticConfig,
    generate_synthetic_panel,
)
from latent_svi_spatial.models.sar import check_logdet_consistency


def test_logdet_identity() -> None:
    data = generate_synthetic_panel(
        SyntheticConfig(
            n=20,
            r=3,
            seed=123,
            normalize_w="none",
            rho=0.01,
        )
    )

    err = check_logdet_consistency(
        data.W,
        data.H,
        data.C,
        float(data.rho),
    )

    assert err < 1e-6