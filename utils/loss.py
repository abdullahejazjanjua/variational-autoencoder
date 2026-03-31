import torch
import torch.nn as nn

class Criterion(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        input_x: torch.Tensor,
        reconstructed_x: torch.Tensor,
        log_var: torch.Tensor,
        mu: torch.Tensor,
    ):
        reconstruction_loss = self.compute_reconstruction_loss(
            input_x=input_x, reconstructed_x=reconstructed_x
        )
        kl_loss = self.compute_kl_loss(log_var=log_var, mu=mu)

        return reconstruction_loss + kl_loss, reconstruction_loss, kl_loss

    def compute_kl_loss(self, log_var: torch.Tensor, mu: torch.Tensor):
        """
        Because the prior p(z) is a standard normal N(0, I) and the posterior q_phi(z|x) is modeled as a 
        Gaussian N(mu, sigma^2) with a diagonal covariance matrix, the KL divergence has a closed-form 
        analytical solution.

        No sampling is required to compute this term. The mathematical formula for a latent vector of 
        dimension J is:
        D_KL = -0.5 * Sum_{j=1 to J} (1 + log(sigma_j^2) - mu_j^2 - sigma_j^2)
        """
        return -0.5 * torch.sum(1 + log_var - mu**2 - torch.exp(log_var))

    def compute_reconstruction_loss(
        self, input_x: torch.Tensor, reconstructed_x: torch.Tensor
    ):
        """
        The connection between the log-likelihood log p(x|z; theta) and the Mean Squared Error comes from 
        treating the decoder's output as the parameters of a probability distribution. Let the decoder 
        network output a vector x_hat, which represents the mean of a Gaussian distribution with a 
        fixed variance, such as sigma^2 = 1.

        For continuous input data x with D dimensions, the probability density function for a single data point 
        given z is the product of the probabilities of its individual dimensions:
        p(x|z; theta) = Product_{i=1 to D} [ (1 / sqrt(2 * pi * sigma^2)) * exp( -(x_i - x_hat_i)^2 / (2 * sigma^2) ) ]

        Taking the natural logarithm of this likelihood turns the product into a sum:
        log p(x|z; theta) = Sum_{i=1 to D} [ -0.5 * log(2 * pi * sigma^2) - (x_i - x_hat_i)^2 / (2 * sigma^2) ]

        During training, the goal is to maximize this log-likelihood with respect to the network parameters 
        theta. The first term, -0.5 * log(2 * pi * sigma^2), is a constant with respect to theta and can 
        be ignored during optimization. Assuming sigma^2 = 1, the equation simplifies to:
        log p(x|z; theta) is proportional to -0.5 * Sum_{i=1 to D} [ (x_i - x_hat_i)^2 ]

        Maximizing this negative squared difference is mathematically identical to minimizing the 
        Mean Squared Error (MSE) between the original image x and the reconstruction x_hat.
        """
        return nn.MSELoss(reduction="sum")(reconstructed_x, input_x)
