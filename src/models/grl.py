import torch
from torch import nn
from torch.autograd import Function

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_grl):
        """
        In the forward pass, GRL acts as identity.
        """
        ctx.lambda_grl = lambda_grl
        # Return input as-is
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass, reverse the gradient by multiplying
        with -lambda_grl.
        """
        return grad_output.neg() * ctx.lambda_grl, None

class GradientReversal(nn.Module):
    """
    A torch.nn.Module wrapper for the gradient reversal operation.
    Usage:
        grl = GradientReversal()
        x_reversed = grl(x, lambda_grl=1.0)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, lambda_grl: float = 1.0):
        return GradientReversalFunction.apply(x, lambda_grl)