import logging
from scipy.stats import norm
import numpy as np

from tlbo.acquisition_function.base_acquisition import BaseAcquisitionFunction

logger = logging.getLogger(__name__)


class EI(BaseAcquisitionFunction):

    def __init__(self, model, par=0.0):

        r"""
        Computes for a given x the expected improvement as
        acquisition_functions value.
        :math:`EI(X) :=
            \mathbb{E}\left[ \max\{0, f(\mathbf{X^+}) -
                f_{t+1}(\mathbf{X}) - \xi\right] \} ]`, with
        :math:`f(X^+)` as the incumbent.

        Parameters
        ----------
        model: Model object
            A model that implements at least
                 - predict(X)
                 - getCurrentBestX().
            If you want to calculate derivatives than it should also support
                 - predictive_gradients(X)

        par: float
            Controls the balance between exploration
            and exploitation of the acquisition_functions function. Default is 0.0
        """

        super(EI, self).__init__(model)
        self.par = par

    def compute(self, X, **kwargs):
        """
        Computes the EI value.

        Parameters
        ----------
        X: np.ndarray(N, D), The input point where the acquisition_functions function
            should be evaluate. The dimensionality of X is (N, D), with N as
            the number of points to evaluate at and D is the number of
            dimensions of one X.

        Returns
        -------
        np.ndarray(N)
            Expected Improvement of X
        """

        m, v = self.model.predict(X)

        # Use the best seen observation as incumbent

        _, eta = self.model.get_incumbent(scaled=self.model.scale)

        s = np.sqrt(v)

        if (s == 0).any():
            f = np.array([[0]])
        else:
            # z = (eta - m - self.par) / s
            # f = s * (z * norm.cdf(z) + norm.pdf(z))
            z = (m - self.par - eta) / s
            f = (m - self.par - eta) * norm.cdf(z) + s * norm.pdf(z)

            if (f < 0).any():
                logger.error("Expected Improvement is smaller than 0!")
                raise ValueError
        return f
