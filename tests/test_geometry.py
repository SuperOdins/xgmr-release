import numpy as np
import torch

from xgmr.geometry.homography import dlt_homography


def test_dlt_identity():
    pts_src = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    pts_dst = pts_src.clone()
    H = dlt_homography(pts_src, pts_dst)
    eye = torch.eye(3)
    assert torch.allclose(H, eye, atol=1e-1)
