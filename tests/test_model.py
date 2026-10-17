import torch

from prism.model import PRESETS, GeodesicTransformer


def small_model(**kw):
    return GeodesicTransformer(dim=64, heads=4, n_anchors=32, n_self=2, n_dec_cross=1, **kw).eval()


def random_cloud(n=256):
    p = torch.randn(1, n, 3)
    p = p / p.norm(dim=-1, keepdim=True)
    return p, p.clone()  # unit sphere, outward normals


def test_forward_shapes():
    m = small_model()
    p, n = random_cloud()
    pairs = torch.randint(0, 256, (1, 100, 2))
    with torch.no_grad():
        d = m(p, n, pairs, anchor_idx=torch.arange(32)[None])
    assert d.shape == (1, 100)


def test_symmetric_and_not_below_euclidean():
    m = small_model()
    p, n = random_cloud()
    pairs = torch.randint(0, 256, (1, 100, 2))
    aidx = torch.arange(32)[None]
    with torch.no_grad():
        d1 = m(p, n, pairs, anchor_idx=aidx)
        d2 = m(p, n, pairs.flip(-1), anchor_idx=aidx)
    assert torch.allclose(d1, d2, atol=1e-5)
    eu = (p[0, pairs[0, :, 0]] - p[0, pairs[0, :, 1]]).norm(dim=-1)
    assert (d1[0] >= eu - 1e-5).all()


def test_distance_matrix_zero_diagonal():
    m = small_model()
    p, n = random_cloud(64)
    D = m.distance_matrix(p[0], n[0])
    assert D.shape == (64, 64)
    assert torch.all(D.diagonal() == 0)


def test_presets_build():
    for cfg in PRESETS.values():
        GeodesicTransformer(**{**cfg, "n_self": 1})


def test_all_params_receive_grad():  # required for DDP without find_unused_parameters
    m = small_model().train()
    p, n = random_cloud()
    pairs = torch.randint(0, 256, (1, 50, 2))
    m(p, n, pairs).sum().backward()
    assert all(q.grad is not None for q in m.parameters())
