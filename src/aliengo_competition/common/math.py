import math

import torch


def quat_conjugate_xyzw(quat: torch.Tensor) -> torch.Tensor:
    result = quat.clone()
    result[..., :3] = -result[..., :3]
    return result


def quat_multiply_xyzw(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    x1, y1, z1, w1 = q1.unbind(dim=-1)
    x2, y2, z2, w2 = q2.unbind(dim=-1)
    return torch.stack(
        (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ),
        dim=-1,
    )


def quat_rotate_inverse_xyzw(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros_like(vec[..., :1])
    pure = torch.cat((vec, zeros), dim=-1)
    return quat_multiply_xyzw(quat_conjugate_xyzw(quat), quat_multiply_xyzw(pure, quat))[..., :3]


def projected_gravity_xyzw(quat: torch.Tensor) -> torch.Tensor:
    gravity = torch.zeros(quat.shape[0], 3, device=quat.device, dtype=quat.dtype)
    gravity[:, 2] = -1.0
    return quat_rotate_inverse_xyzw(quat, gravity)


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi
