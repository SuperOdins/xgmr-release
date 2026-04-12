import torch

from xgmr.models.matcher import dual_softmax_match
from xgmr.losses import matching_nll


def test_matching_loss_decreases():
    torch.manual_seed(0)
    S = torch.randn(1, 16, 16, requires_grad=True)
    optimizer = torch.optim.SGD([S], lr=0.1)
    target = torch.eye(16).unsqueeze(0)
    losses = []
    for _ in range(5):
        optimizer.zero_grad()
        P = dual_softmax_match(S)
        loss = matching_nll(P, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]
