import torch
from xgmr.models.self_calib import SelfCalibratingHead
from xgmr.config import XGMRConfig
from xgmr.utils.ablation import set_ablation_config

def test_selfcalib_ablation():
    cfg = XGMRConfig()
    set_ablation_config(cfg)
    
    dim = 64
    model = SelfCalibratingHead(dim=dim, iters=1)
    
    # Manually set a bias so baseline is NOT identity
    with torch.no_grad():
        model.fc[-1].bias.fill_(0.1)
    
    f_rgb = torch.randn(1, dim, 8, 8)
    f_t = torch.randn(1, dim, 8, 8)
    matches = {} # dummy matches
    
    # CASE 1: Baseline (Ablation off)
    out_base = model(f_rgb, f_t, matches)
    diff_base = (out_base["H"].cpu() - torch.eye(3)).abs().mean()
    print(f"Base H Diff: {diff_base.item():.6f}")
    assert diff_base > 1e-3, "Baseline should NOT be identity"
    
    # CASE 2: Ablation (no_selfcalib=True)
    cfg.ablation.no_selfcalib = True
    out_abl = model(f_rgb, f_t, matches)
    
    # Expected ablation result: Identity homography and original thermal features
    # H should be identity
    H_expected = torch.eye(3).unsqueeze(0)
    diff_h = (out_abl["H"].cpu() - H_expected).abs().mean()
    print(f"H Diff: {diff_h.item():.6f}")
    assert diff_h < 1e-6, "Ablation should return identity homography"
    
    # warped_t should be original t
    diff_t = (out_abl["warped_t"] - f_t).abs().mean()
    print(f"T Diff: {diff_t.item():.6f}")
    assert diff_t < 1e-6, "Ablation should return unwarped thermal features"
    
    print("✅ Self-Calibration Ablation Test Passed!")

if __name__ == "__main__":
    test_selfcalib_ablation()
