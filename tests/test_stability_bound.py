from latent_svi_spatial.data.synthetic import (
    SyntheticConfig,
    generate_synthetic_panel,
    stable_rho_upper_bound,
)


def test_stability_bound() -> None:
    data = generate_synthetic_panel(
        SyntheticConfig(
            n=20,
            r=3,
            seed=123,
            normalize_w="none",
            rho=0.01,
        )
    )

    upper = stable_rho_upper_bound(data.W)

    assert abs(float(data.rho)) < upper