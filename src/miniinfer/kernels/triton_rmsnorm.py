"""One Triton program per row: reduce, normalize, scale, and store."""

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm(X, W, Y, WIDTH: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    column = tl.arange(0, BLOCK)
    valid = column < WIDTH
    # Padding contributes zero to the sum; divide by the real width.
    values = tl.load(X + row * WIDTH + column, mask=valid, other=0).to(tl.float32)
    weights = tl.load(W + column, mask=valid, other=0).to(tl.float32)
    mean_square = tl.sum(values * values, axis=0) / WIDTH
    normalized = values * tl.rsqrt(mean_square + EPS)
    tl.store(Y + row * WIDTH + column, normalized * weights, mask=valid)


def launch(x, weight, eps):
    width = x.shape[-1]
    block = triton.next_power_of_2(width)
    warps = 4 if block <= 2048 else 8
    output = torch.empty_like(x)
    _rmsnorm[(x.numel() // width,)](
        x, weight, output, WIDTH=width, EPS=eps, BLOCK=block, num_warps=warps,
    )
    return output
