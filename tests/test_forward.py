import torch

from xgmr.models import XGMR


def test_forward():
    backbone_cfg = {"name": "light", "base_dim": 16, "depth": 1, "in_channels": 64}
    mba_cfg = {"in_ch_rgb": 3, "in_ch_t": 1, "out_dim": 64, "use_eaef": True}
    matcher_cfg = {"dim": 64, "layers": (1, 1), "matcher": "dual_softmax"}
    selfcalib_cfg = {"dim": 64, "tps_grid": 3, "iters": 1, "tps_reg": 1e-3}
    qfusion_cfg = {"tile": 16, "k": 8.0, "tau": 0.5, "mode": "feature"}

    model = XGMR(backbone_cfg, mba_cfg, matcher_cfg, selfcalib_cfg, qfusion_cfg)
    rgb = torch.rand(1, 3, 128, 128)
    thermal = torch.rand(1, 1, 128, 128)
    out = model(rgb, thermal)

    assert "matches_c" in out
    assert "matches_f" in out
    assert "H" in out
    assert "warped_t" in out
    assert "Qmap" in out
