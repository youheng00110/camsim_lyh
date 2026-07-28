import torch

from dwm.models.urope.urope import URoPEDotProductAttention


def test_urope_forward_and_backward():
    batch_size = 1
    camera_count = 2
    patch_height = 2
    patch_width = 3
    head_count = 4
    head_dim = 8
    token_count = camera_count * patch_height * patch_width

    query = torch.randn(
        batch_size,
        head_count,
        token_count,
        head_dim,
        requires_grad=True,
    )
    key = torch.randn_like(query, requires_grad=True)
    value = torch.randn_like(query, requires_grad=True)

    viewmats = torch.eye(4).reshape(1, 1, 4, 4).repeat(
        batch_size,
        camera_count,
        1,
        1,
    )
    viewmats[:, 1, 0, 3] = -1.0

    intrinsics = torch.eye(3).reshape(1, 1, 3, 3).repeat(
        batch_size,
        camera_count,
        1,
        1,
    )
    intrinsics[..., 0, 0] = 2.0
    intrinsics[..., 1, 1] = 2.0
    intrinsics[..., 0, 2] = patch_width / 2
    intrinsics[..., 1, 2] = patch_height / 2

    attention = URoPEDotProductAttention(
        head_num=head_count,
        head_dim=head_dim,
        group_size=2,
        camera_convention="opencv",
    )
    output = attention(
        query,
        key,
        value,
        viewmats=viewmats,
        intrinsics=intrinsics,
        patch_width=patch_width,
        patch_height=patch_height,
    )

    assert output.shape == query.shape
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert query.grad is not None
    assert key.grad is not None
    assert value.grad is not None
