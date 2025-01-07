from typing import Optional, Union

from botorch.exceptions import BotorchTensorDimensionError
from botorch.models.transforms.input import ReversibleInputTransform
from botorch.models.transforms.utils import subset_transform
from torch import Tensor
import torch


class MatrixTransform(ReversibleInputTransform, torch.nn.Module):
    def __init__(
            self,
            d: int,
            transformation_matrix: Tensor,
            indices: Optional[Union[list[int], Tensor]] = None,
            batch_shape: torch.Size = torch.Size(),  # noqa: B008
            transform_on_train: bool = True,
            transform_on_eval: bool = True,
            transform_on_fantasize: bool = True,
            reverse: bool = False,
    ) -> None:
        r"""Apply affine transformation to input:

            `output = (input - offset) / coefficient`

        Args:
            d: The dimension of the input space.
            transformation_matrix: Transformation tensor, shape must to be
                broadcastable with `(batch_shape x n x d)`-dim input tensors.
            indices: The indices of the inputs to transform. If omitted,
                take all dimensions of the inputs into account. Either a list of ints
                or a Tensor of type `torch.long`.
            batch_shape: The batch shape of the inputs (assuming input tensors
                of shape `batch_shape x n x d`). If provided, perform individual
                transformation per batch, otherwise uses a single transformation.
            transform_on_train: A boolean indicating whether to apply the
                transform in train() mode. Default: True.
            transform_on_eval: A boolean indicating whether to apply the
                transform in eval() mode. Default: True.
            transform_on_fantasize: A boolean indicating whether to apply the
                transform when called from within a `fantasize` call. Default: True.
            reverse: A boolean indicating whether the forward pass should untransform
                the inputs.
        """
        super().__init__()
        if (indices is not None) and (len(indices) == 0):
            raise ValueError("`indices` list is empty!")
        if (indices is not None) and (len(indices) > 0):
            indices = torch.as_tensor(
                indices, dtype=torch.long, device=transformation_matrix.device
            )
            if len(indices) > d:
                raise ValueError("Can provide at most `d` indices!")
            if (indices > d - 1).any():
                raise ValueError("Elements of `indices` have to be smaller than `d`!")
            if len(indices.unique()) != len(indices):
                raise ValueError("Elements of `indices` tensor must be unique!")
            self.register_buffer("indices", indices)

        self._d = d
        self.register_buffer("transformation_matrix", transformation_matrix)
        self.batch_shape = batch_shape
        self.transform_on_train = transform_on_train
        self.transform_on_eval = transform_on_eval
        self.transform_on_fantasize = transform_on_fantasize
        self.reverse = reverse

    @subset_transform
    def _transform(self, X: Tensor) -> Tensor:
        r"""Apply affine transformation to input.

        Args:
            X: A `batch_shape x n x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x n x d`-dim tensor of transformed inputs.
        """
        self._check_shape(X)
        self._to(X)
        return (self.transformation_matrix @ X.transpose(-1, -2)).transpose(-1, -2)

    @subset_transform
    def _untransform(self, X: Tensor) -> Tensor:
        r"""Apply inverse of affine transformation.

        Args:
            X: A `batch_shape x n x d`-dim tensor of transformed inputs.

        Returns:
            A `batch_shape x n x d`-dim tensor of un-transformed inputs.
        """
        self._to(X)
        return (
            self.transformation_matrix.inverse() @ X.transpose(-1, -2)
        ).transpose(-1, -2)

    def _check_shape(self, X: Tensor) -> None:
        """Checking input dimensions, included to increase code sharing
        among the derived classes Normalize and InputStandardize.
        """
        if X.size(-1) != self.transformation_matrix.size(-1):
            raise BotorchTensorDimensionError(
                f"Wrong input dimension. Received {X.size(-1)}, "
                f"expected {self.transformation_matrix.size(-1)}."
            )
        if X.ndim < 2:
            raise BotorchTensorDimensionError(
                f"`X` must have at least 2 dimensions, but has {X.ndim}."
            )

        n = len(self.batch_shape) + 2
        if self.training and X.ndim < n:
            raise ValueError(
                f"`X` must have at least {n} dimensions, {n - 2} batch and 2 innate"
                f" , but has {X.ndim}."
            )


    def _to(self, X: Tensor) -> None:
        r"""Makes coefficient and offset have same device and dtype as X."""
        self.transformation_matrix = self.transformation_matrix.to(X)


