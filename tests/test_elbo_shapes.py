from latent_svi_spatial.data.synthetic import SyntheticConfig, generate_synthetic_panel
from latent_svi_spatial.vi.elbo import estimate_elbo
from latent_svi_spatial.vi.variational_family import VariationalFamily


def test_elbo_shapes_and_finiteness() -> None:
    data = generate_synthetic_panel(
        SyntheticConfig(
            n=20,
            t=5,
            p=3,
            r=3,
            seed=123,
            normalize_w="none",
            rho=0.01,
        )
    )

    _, _, p_eff = data.X.shape

    vf = VariationalFamily(
        n=data.H.shape[0],
        r=data.H.shape[1],
        p_eff=p_eff,
        rho_max=0.05,
    )

    result = estimate_elbo(
        variational_family=vf,
        X=data.X,
        y=data.y,
        n_mc_samples=2,
    )

    assert result.elbo.isfinite()
    assert result.loss.isfinite()
    assert result.expected_log_likelihood.isfinite()
    assert result.kl_total.isfinite()