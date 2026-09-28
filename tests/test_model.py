# cnn building blocks: network shape and reach, features, weak-label rules, patches, scoring.

import numpy as np
import pandas as pd
import pytest
import torch

from src.model.features import channel_names, shift_month
from src.model.metrics import scores, weighted_accuracy
from src.model.net import PixelFCN, receptive_field
from src.model.samples import extract_patches, weak_rules


def test_network_output_shrinks_by_its_reach():
    net = PixelFCN(in_ch=14, n_classes=5).eval()
    assert net(torch.zeros(2, 14, 37, 41)).shape == (2, 5, 37 - 22, 41 - 22)
    assert net(torch.zeros(1, 14, 23, 23)).shape == (1, 5, 1, 1)   # training patch -> 1 pixel


def test_receptive_field_matches_what_the_network_sees():
    # poke one input pixel and see how far the output changes: that's the network's reach
    net = PixelFCN(in_ch=1, n_classes=2).eval()
    x = torch.zeros(1, 1, 61, 61)
    base = net(x)
    x[0, 0, 30, 30] = 5.0
    changed = (net(x) - base).abs().sum(1)[0] > 1e-6
    rows = torch.nonzero(changed.any(1)).flatten()
    assert rows.max() - rows.min() + 1 == receptive_field() == 23


def test_whole_image_prediction_matches_patch_prediction():
    # mapping (padded whole image, in tiles) must give the same answer as the training setup
    # (one 23 x 23 patch per pixel), or the maps wouldn't show what the model learned
    import scripts.predict_cnn as pc
    torch.manual_seed(0)
    net = PixelFCN(in_ch=3, n_classes=4).eval()
    img = np.random.default_rng(0).normal(size=(3, 40, 50)).astype("float32")
    pc.TILE = 16  # force several tiles
    prob = pc.predict_image(net, img, "cpu")
    padded = np.pad(img, ((0, 0), (11, 11), (11, 11)))
    for r, c in [(0, 0), (20, 25), (39, 49), (15, 16)]:
        patch = torch.from_numpy(padded[None, :, r:r + 23, c:c + 23].copy())
        p = torch.softmax(net(patch), 1)[0, :, 0, 0].detach().numpy()
        np.testing.assert_allclose(prob[:, r, c], p, atol=1e-5)


def test_channel_names():
    assert len(channel_names()) == 14
    assert len(channel_names((-1, -2))) == 14 + 2 * 12


@pytest.mark.parametrize("month, off, expected", [
    ("2024-03", -1, "2024-02"), ("2024-01", -1, "2023-12"), ("2023-11", 3, "2024-02")])
def test_shift_month(month, off, expected):
    assert shift_month(month, off) == expected


def spec(green, nir, ndvi, mndwi):
    """[12, 1, 1] spectral layers with the values the weak rules look at."""
    s = np.zeros((12, 1, 1), "float32")
    s[1], s[3], s[10], s[11] = green, nir, ndvi, mndwi
    return s


def test_weak_rules_only_fire_when_sure():
    assert weak_rules(spec(0.05, 0.03, -0.3, 0.6), "2024-07")["water"].all()
    assert not weak_rules(spec(0.05, 0.03, -0.1, 0.1), "2024-07")["water"].any()   # borderline
    assert weak_rules(spec(0.2, 0.25, 0.1, -0.3), "2024-07")["bare"].all()
    assert weak_rules(spec(0.05, 0.4, 0.8, -0.4), "2024-07")["dense_veg"].all()
    assert not weak_rules(spec(0.05, 0.3, 0.5, -0.4), "2024-07")["dense_veg"].any()  # could be sparse
    assert "sparse_veg" not in weak_rules(spec(0.05, 0.3, 0.4, -0.4), "2024-07")     # never weak


def test_snow_weak_labels_only_in_winter():
    s = spec(0.6, 0.55, 0.0, 0.7)
    assert weak_rules(s, "2025-01")["snow_ice"].all()
    assert "snow_ice" not in weak_rules(s, "2025-07")


def test_patches_are_centred_and_zero_padded():
    feats = np.arange(2 * 10 * 10, dtype="float32").reshape(2, 10, 10)
    samples = pd.DataFrame({"month": ["m", "m"], "row": [5, 0], "col": [5, 0]})
    p = extract_patches(samples, lambda m: (feats, None), half=2)
    assert p.shape == (2, 2, 5, 5)
    assert p[0, 1, 2, 2] == feats[1, 5, 5]
    assert p[1, 0, 2, 2] == feats[0, 0, 0] and p[1, 0, 0, 0] == 0   # outside the image


def test_scores_count_every_class_equally():
    y = ["water"] * 8 + ["sparse_veg"] * 2
    p = ["water"] * 10
    s = scores(y, p)
    assert s["accuracy"] == 0.8
    assert s["per_class_f1"]["sparse_veg"] == 0
    assert s["macro_f1"] == pytest.approx((16 / 18 + 0) / 2)


def test_weighted_accuracy_undoes_oversampling():
    # stratum a is 90% of the area but only half the points
    y = ["x"] * 4
    p = ["x", "x", "y", "y"]
    strata = ["a", "a", "b", "b"]
    assert weighted_accuracy(y, p, strata, {"a": 0.9, "b": 0.1}) == pytest.approx(0.9)


def test_weak_sparse_only_when_asked():
    s = spec(0.05, 0.3, 0.4, -0.4)
    assert "sparse_veg" not in weak_rules(s, "2024-07")
    assert weak_rules(s, "2024-07", weak_sparse=True)["sparse_veg"].all()
    assert not weak_rules(spec(0.05, 0.3, 0.3, -0.4), "2024-07", True)["sparse_veg"].any()  # edge


def test_smaller_reach_is_configurable():
    net = PixelFCN(in_ch=4, n_classes=5, dilations=(1, 1, 1, 1, 1)).eval()
    assert net.rf == 11
    assert net(torch.zeros(1, 4, 11, 11)).shape == (1, 5, 1, 1)


def test_speckle_rate():
    from src.stats import speckle_rate
    flat = np.full((10, 10), 2)
    noisy = flat.copy()
    noisy[5, 5] = 3                                  # one odd pixel out of 100
    everything = np.ones((10, 10), bool)
    assert speckle_rate(flat, everything) == 0
    assert speckle_rate(noisy, everything) == pytest.approx(0.01)


def test_stratified_area_estimate():
    from src.model.metrics import stratified_area_estimate
    # stratum a: 900 px, half its points water; stratum b: 100 px, all bare
    labels = ["water", "bare"] * 10 + ["bare"] * 10
    strata = ["a"] * 20 + ["b"] * 10
    est = stratified_area_estimate(labels, strata, {"a": 900, "b": 100}, px_km2=1,
                                   classes=["water", "bare"]).set_index("class")
    assert est.loc["water", "km2"] == pytest.approx(450)          # 900 * 0.5
    assert est.loc["bare", "km2"] == pytest.approx(450 + 100)
    assert est.loc["water", "ci95_low"] < 450 < est.loc["water", "ci95_high"]
    # all-bare stratum b is certain, so the uncertainty comes from stratum a only
    assert (est.loc["water", "ci95_high"] - 450) == pytest.approx(est.loc["bare", "ci95_high"] - 550)
