import torch
from xgmr.models.qfusion import QFusion
from xgmr.config import XGMRConfig
from xgmr.utils.ablation import set_ablation_config

def test_qfusion_ablation():
    cfg = XGMRConfig()
    set_ablation_config(cfg)
    
    dim = 64
    model = QFusion(dim=dim)
    
    F_rgb = torch.ones(1, dim, 8, 8)
    F_t = torch.zeros(1, dim, 8, 8) # All black thermal
    match_stats = {"num_matches": 100, "inlier_ratio": 0.8, "prob_mean": 0.5, "reproj_mean": 0.1}
    
    # CASE 1: Baseline (Ablation off)
    # Since thermal is zero, and matches are good, QFusion should favor Thermal or mixed depending on stats
    fused_base, _ = model(F_rgb, F_t, match_stats)
    
    # CASE 2: Ablation (no_qfusion=True)
    cfg.ablation.no_qfusion = True
    fused_abl, aux_abl = model(F_rgb, F_t, match_stats)
    
    # Expected ablation result: simple average (0.5 * 1 + 0.5 * 0 = 0.5)
    expected = (F_rgb + F_t) / 2.0
    
    diff = (fused_abl - expected).abs().mean()
    print(f"Diff: {diff.item():.4f}")
    assert diff < 1e-6, "Ablation should return simple average"
    assert aux_abl["quality_scalar"] == 0.0
    
    print("✅ Q-Fusion Ablation Test Passed!")

if __name__ == "__main__":
    test_qfusion_ablation()
