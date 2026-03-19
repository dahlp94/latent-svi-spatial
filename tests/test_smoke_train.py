from latent_svi_spatial.data.synthetic import SyntheticConfig, generate_synthetic_panel
from latent_svi_spatial.train.trainer import TrainConfig, train_variational_model
from latent_svi_spatial.vi.variational_family import VariationalFamily


def test_smoke_train() -> None:
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

    history = train_variational_model(
        variational_family=vf,
        X=data.X,
        y=data.y,
        config=TrainConfig(
            n_steps=5,
            lr=1e-2,
            n_mc_samples=1,
            log_every=1,
            verbose=False,
        ),
    )

    assert len(history.records) == 5
    assert history.last() is not None
    assert history.last().elbo == history.last().elbo  # not NaN