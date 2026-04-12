import torch
from xgmr.config import XGMRConfig
from xgmr.models.xgmr import XGMR
from xgmr.utils.ablation import set_ablation_config

def test_final_integration():
    cfg = XGMRConfig()
    set_ablation_config(cfg)
    
    # Initialize full model
    model = XGMR(
        cfg.model.backbone,
        cfg.model.mba,
        cfg.model.matcher,
        cfg.model.selfcalib,
        cfg.model.qfusion,
        use_mba=True,
        use_self_calib=True,
        use_qfusion=True
    )
    model.eval()
    
    # Mock inputs
    rgb = torch.randn(1, 3, 256, 256)
    thermal = torch.randn(1, 1, 256, 256)
    
    print("Testing ALL Ablations ON (Vanilla Mode)...")
    cfg.ablation.no_eba = True
    cfg.ablation.no_lgeo = True
    cfg.ablation.no_mba = True
    cfg.ablation.no_qfusion = True
    cfg.ablation.no_selfcalib = True
    
    with torch.no_grad():
        out = model(rgb, thermal)
    
    # Check H is identity
    H_id = torch.eye(3).unsqueeze(0)
    assert (out["H"].cpu() - H_id).abs().mean() < 1e-6
    
    # Check Qmap is zero
    assert out["Qmap"].abs().mean() == 0.0
    
    print("Testing ALL Ablations OFF (Full Mode)...")
    cfg.ablation.no_eba = False
    cfg.ablation.no_lgeo = False
    cfg.ablation.no_mba = False
    cfg.ablation.no_qfusion = False
    cfg.ablation.no_selfcalib = False
    
    with torch.no_grad():
        out_full = model(rgb, thermal)
    
    # In full mode, H should likely NOT be exactly identity (due to random weights and matches)
    # But Qmap should NOT be exactly zero
    assert out_full["Qmap"].abs().mean() > 0.0 or out_full["quality_scalar"] != 0.0
    
    print("✅ Final Integration Test Passed!")

if __name__ == "__main__":
    test_final_integration()
