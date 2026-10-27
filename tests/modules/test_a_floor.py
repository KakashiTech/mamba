import torch

from mamba_ssm.modules.a_floor import apply_a_floor


def test_a_floor_clamps_near_zero():
    A = torch.tensor([-1e-8, -1e-5, -1e-4, -1.0])
    expected = torch.tensor([-1e-4, -1e-4, -1e-4, -1.0])
    assert torch.equal(apply_a_floor(A, 1e-4), expected)


def test_a_floor_is_noop_when_already_below_floor():
    A = torch.tensor([-1e-3, -1.0])
    assert torch.equal(apply_a_floor(A, 1e-4), A)


def test_a_floor_can_be_disabled():
    A = torch.tensor([-1e-8, -1.0])
    assert torch.equal(apply_a_floor(A, None), A)
