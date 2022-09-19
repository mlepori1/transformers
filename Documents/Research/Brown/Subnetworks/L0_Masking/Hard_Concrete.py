import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class HardConcrete(nn.Module):
    """A HarcConcrete module.

    Use this module to create a mask of size N, which you can
    then use to perform L0 regularization. Note that in general,
    we also provide utilities which introduce HardConrete modules
    in the desired places in your model. See ``utils`` for details.

    To obtain a mask, simply run a forward pass through the module
    with no input data. The mask is sampled in training mode, and
    fixed during evaluation mode:

    >>> module = HardConcrete(n_in=100)
    >>> mask = module()
    >>> norm = module.l0_norm()

    """

    def __init__(self,
                 n_in: int,
                 n_out: int,
                 init_mean: float = 0.5,
                 init_std: float = 0.01,
                 temperature: float = 2./3.,
                 stretch: float = 0.1,
                 eps: float = 1e-6) -> None:
        """Initialize the HardConcrete module.

        Parameters
        ----------
        n_in : int
            The number of hard concrete variables in dim 0 of this mask.
        n_out : int
            The number of hard concrete variables in dim 1 of this mask.
        init_mean : float, optional
            Initialization value for hard concrete parameter,
            by default 0.5.,
        init_std: float, optional
            Used to initialize the hard concrete parameters,
            by default 0.01.
        temperature : float, optional
            Temperature used to control the sharpness of the
            distribution, by default 1.0
        stretch : float, optional
            Stretch the sampled value from [0, 1] to the interval
            [-stretch, 1 + stretch], by default 0.1.

        """
        super().__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.limit_l = -stretch
        self.limit_r = 1.0 + stretch
        self.log_alpha = nn.Parameter(torch.zeros(n_in, n_out))  # type: ignore
        self.beta = temperature
        self.init_mean = init_mean
        self.init_std = init_std
        self.bias = -self.beta * math.log(-self.limit_l / self.limit_r)

        self.eps = eps
        self.compiled_mask = None
        self.reset_parameters()

    def train(self, train_bool):
        self.training=train_bool

    def reset_parameters(self):
        """Reset the parameters of this module."""
        self.compiled_mask = None
        mean = math.log(1 - self.init_mean) - math.log(self.init_mean)
        self.log_alpha.data.normal_(mean, self.init_std)

    def l0_norm(self) -> torch.Tensor:
        """Compute the expected L0 norm of this mask.

        Returns
        -------
        torch.Tensor
            The expected L0 norm.

        """
        return (self.log_alpha + self.bias).sigmoid().sum()

    def forward(self) -> torch.Tensor:  # type: ignore
        """Sample a harconcrete mask.

        Returns
        -------
        torch.Tensor
            The sampled binary mask

        """
        if self.training:
            # Reset the compiled mask
            self.compiled_mask = None
            # Sample mask dynamically
            u = self.log_alpha.new_empty((self.n_in, self.n_out)).uniform_(self.eps, 1 - self.eps)  # type: ignore
            s = F.sigmoid((torch.log(u / (1 - u)) + self.log_alpha) / self.beta)
            s = s * (self.limit_r - self.limit_l) + self.limit_l
            mask = s.clamp(min=0., max=1.)

        else:
            # Compile new mask if not cached
            if self.compiled_mask is None:
                # Sample a mask, assign all values to 1 or 0, cache it
                s = F.sigmoid(self.log_alpha)
                s = s * (self.limit_r - self.limit_l) + self.limit_l
                mask = s.clamp(min=0., max=1.)
                # Discretize the mask
                ones = mask > .5
                zeros = ~ones
                mask[zeros] = 0.
                mask[ones] = 1.
                self.compiled_mask = mask
            mask = self.compiled_mask

        return mask

    def extre_repr(self) -> str:
        return str(self.n_in), str(self.n_out)

    def __repr__(self) -> str:
        return "{}({})".format(self.__class__.__name__, self.extre_repr())
