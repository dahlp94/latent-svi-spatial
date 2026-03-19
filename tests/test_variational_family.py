from latent_svi_spatial.vi.variational_family import VariationalFamily


def test_variational_sampling():
    n, r, p = 10, 3, 4

    vf = VariationalFamily(n=n, r=r, p_eff=p)

    sample = vf.rsample()

    # ---- shape checks ----
    assert sample.H.shape == (n, r)
    assert sample.C.shape == (r, r)
    assert sample.beta.shape == (p,)
    assert isinstance(sample.rho.item(), float)
    assert isinstance(sample.sigma2.item(), float)

    # ---- constraints ----
    # H rows sum to 1
    row_sums = sample.H.sum(dim=1)
    assert ((row_sums - 1.0).abs() < 1e-6).all()

    # C positive
    assert (sample.C > 0).all()

    # rho in (0, rho_max)
    assert 0.0 < sample.rho < vf.rho_max

    # sigma2 positive
    assert sample.sigma2 > 0

    # ---- KL finite ----
    kl = vf.kl_total()
    assert kl.isfinite()