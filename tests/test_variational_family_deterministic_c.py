from latent_svi_spatial.vi.variational_family import VariationalFamily


def test_variational_family_deterministic_c() -> None:
    vf = VariationalFamily(
        n=10,
        r=3,
        p_eff=4,
        deterministic_C=True,
    )

    sample = vf.rsample()

    assert sample.H.shape == (10, 3)
    assert sample.C.shape == (3, 3)
    assert sample.beta.shape == (4,)
    assert sample.sigma2 > 0
    assert 0.0 < sample.rho < vf.rho_max

    # deterministic C => KL_C should be zero
    assert vf.kl_C().item() == 0.0