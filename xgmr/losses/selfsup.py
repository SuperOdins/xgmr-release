"""Self-supervised loss primitives for XGMR."""

from __future__ import annotations

import torch
import torch.nn.functional as F




def loss_equivariance(H_student: torch.Tensor, H_synth: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """
    동변성 손실 (Equivariance Loss).
    Korean: 기하학적 일관성을 유지하기 위해, 이미지 변환 $S$가 적용된 상태에서 추정된 호모그래피가 원래 상태의 추정과 호환되는지 검증한다.
    $\mathcal{L}_{equiv} = \|\mathbf{S}^{-1} \mathbf{H}_{synth} - \mathbf{H}_{student}\|_1$
    English: Verifies geometric consistency by checking if the homography estimated under transform $S$ is compatible with the original estimation.
    $\mathcal{L}_{equiv} = \|\mathbf{S}^{-1} \mathbf{H}_{synth} - \mathbf{H}_{student}\|_1$
    """
    # Inverse computation for $S$ (requires float32 for stability)
    S_f32 = S.to(torch.float32)
    S_inv = torch.inverse(S_f32).to(S.dtype)
    
    # 합성된 호모그래피의 역변환 적용 (Apply inverse transform to synthetic homography)
    composed = torch.bmm(S_inv, H_synth)
    return F.l1_loss(composed, H_student)


from xgmr.utils.ablation import ablation_entry


@ablation_entry("no_lgeo")
def loss_geo(attn_logits: torch.Tensor, epi_bias_mask: torch.Tensor | None) -> torch.Tensor:
    """
    기하학적 제약 손실 (Geometric Constraint Loss).
    Korean: 어텐션 분포가 에피폴라 기하 범위 $M_{epi}$ 내에 집중되도록 유도하여 물리적으로 유효한 정합을 학습한다.
    $\mathcal{L}_{geo} = 1 - \frac{1}{N} \sum_{i} \sum_{j} (P_{i,j} \times M_{i,j}^{epi})$
    English: Guides the attention distribution to concentrate within the epipolar geometric range $M_{epi}$ for physically valid matching.
    $\mathcal{L}_{geo} = 1 - \frac{1}{N} \sum_{i} \sum_{j} (P_{i,j} \times M_{i,j}^{epi})$
    """

    if epi_bias_mask is None:
        return torch.tensor(0.0, device=attn_logits.device, dtype=attn_logits.dtype)

    probs = F.softmax(attn_logits, dim=-1)
    # 마스크 내부의 확률 질량 합 계산 (Sum of probability mass inside the mask)
    inside_mass = (probs * epi_bias_mask).sum(dim=-1)
    return (1.0 - inside_mass).mean()


def loss_geo_ablation(attn_logits: torch.Tensor, epi_bias_mask: torch.Tensor | None) -> torch.Tensor:
    """
    Korean: 기하학적 제약 손실을 비활성화하여 0을 반환한다.
    English: Disables geometric constraint loss by returning zero.
    """
    return torch.tensor(0.0, device=attn_logits.device, dtype=attn_logits.dtype)


def loss_entropy(P: torch.Tensor) -> torch.Tensor:
    """
    정합 엔트로피 정규화 (Matching Entropy Regularization).
    Korean: 정합 확률 행렬 $P$의 엔트로피를 낮추어 모델이 특정 지점에 대해 더욱 명확하고 날카로운 대응 관계를 갖도록 유도한다.
    $\mathcal{L}_{ent} = -\frac{1}{N} \sum_{i,j} P_{i,j} \log P_{i,j}$
    English: Encourages more distinct and sharp correspondence by minimizing the entropy of the matching probability matrix $P$.
    $\mathcal{L}_{ent} = -\frac{1}{N} \sum_{i,j} P_{i,j} \log P_{i,j}$
    """

    eps = 1e-9
    probs = P.clamp(min=eps)
    entropy = -(probs * probs.log()).sum(dim=-1)
    return entropy.mean()

