import torch
from torch import Tensor


def _rotation_matrix() -> Tensor:
    theta = 2 * torch.pi * torch.rand(1)
    phi = torch.acos(2 * torch.rand(1) - 1)
    psi = 2 * torch.pi * torch.rand(1)
    Rz = torch.tensor(
        [
            [torch.cos(theta), -torch.sin(theta), 0],
            [torch.sin(theta), torch.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    Ry = torch.tensor(
        [
            [torch.cos(phi), 0, torch.sin(phi)],
            [0, 1, 0],
            [-torch.sin(phi), 0, torch.cos(phi)],
        ]
    )
    Rx = torch.tensor(
        [
            [1, 0, 0],
            [0, torch.cos(psi), -torch.sin(psi)],
            [0, torch.sin(psi), torch.cos(psi)],
        ]
    )
    return torch.mm(Rz, torch.mm(Ry, Rx))


def sample_fibonacci_grid(
    centroid: Tensor,
    radius: Tensor,
    num_points: int,
    random_rotations: bool = True,
) -> Tensor:
    if num_points < 1:
        raise ValueError(f"num_points must be >= 1, got {num_points}.")
    device = centroid.device
    golden = (1.0 + torch.sqrt(torch.tensor(5.0, device=device))) / 2.0
    idx = torch.arange(num_points, device=device).float()
    theta = 2 * torch.pi * idx / golden
    phi = torch.acos(1 - 2 * (idx + 0.5) / num_points)
    pts = torch.stack(
        [
            radius * torch.sin(phi) * torch.cos(theta),
            radius * torch.sin(phi) * torch.sin(theta),
            radius * torch.cos(phi),
        ],
        dim=1,
    )
    if random_rotations:
        pts = torch.mm(pts, _rotation_matrix().to(device).T)
    return centroid + pts
