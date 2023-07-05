import torch
import torch.nn as nn
import numpy as np


class DDPM(nn.Module):
    """
        DDPM: Denoising Diffusion Probabilistic Model
        
        https://arxiv.org/abs/2006.11239

        Attributes
        ----------
        nn_model    : nn.Module
            the denoising neural network as a pytorch model
        beta1       : float
            the variance for step 1 (default is 0.0001)
        beta2       : float
            the variance for step T (default is 0.02)
        n_timesteps : int
            number of timesteps for diffusion (default is 400)
        device      : str
            the device to be used (default is cuda)
        drop_prob   : float
            dropout for the unconditional generation during training (default 0.1)

        Methods
        -------
        ddpm_schedule(beta1, beta2, n_timesteps)
            Helper method to return pre-computed schedules for DDPM training and sampling process.
        forward(y, c)
            Algorithm (1) for jointly training a diffusion model with classifier-free guidance
        sample(n_sample, size, n_classes, g_weight)
            Algorithm (2) for conditional sampling with classifier-free diffusion guidance
        """

    def __init__(self, nn_model: nn.Module, beta1: float = 0.0001, beta2: float = 0.02, n_timesteps: int = 400,
                 device: str = 'cuda', drop_prob: float = 0.1):
        super(DDPM, self).__init__()
        """
            Parameters
            ----------
            nn_model    : nn.Module
                the denoising neural network as a pytorch model 
            beta1       : float
                the variance for step 1 (default is 0.0001)
            beta2       : float
                the variance for step T (default is 0.02)  
            n_timesteps : int
                number of timesteps for diffusion (default is 400)
            device      : str
                the device to be used (default is cuda)
            drop_prob   : float
                dropout for the unconditional generation during training (default 0.1)
        """

        self.nn_model = nn_model.to(device)

        self.n_timesteps = n_timesteps
        self.device = device
        self.drop_prob = drop_prob

        # register the precomputed dictionary by ddpm_schedule in the buffer
        # e.g., you can access self.sqrt_alpha_bar later. This reduces time for training and inference
        for key, value in self.ddpm_schedule(beta1, beta2, n_timesteps).items():
            self.register_buffer(key, value)

    def ddpm_schedule(self, beta1, beta2, n_timesteps):

        """
            Method to return pre-computed schedules for DDPM training and sampling process.

            Parameters
            ----------
            beta1       : float
                the variance for step 1 (default is 0.0001)
            beta2       : float
                the variance for step T (default is 0.02)
            n_timesteps : int
                number of timesteps for diffusion (default is 400)
            Returns
            ----------
            dictionary of predefined variables for scheduler : dict
        """

        assert beta1 < beta2 < 1.0

        # --- YOUR CODE HERE ---#
        # TODO

        return {
            "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
            "alpha_t": alpha_t,  # \alpha_t
            "alpha_bar_t": alpha_bar_t,  # \bar{\alpha_t}
            "one_over_sqrt_alpha_t": one_over_sqrt_alpha_t,  # 1/\sqrt{\alpha_t}
            "sqrt_alpha_bar_t": sqrt_alpha_bar_t,  # \sqrt{\bar{\alpha_t}}
            "sqrt_one_minus_alpha_bar_t": sqrt_one_minus_alpha_bar_t,  # \sqrt{1-\bar{\alpha_t}}
            "one_minus_alpha_t_over_sqrt_1_minus_alpha_bar_t": one_minus_alpha_t_over_sqrt_1_minus_alpha_bar_t,
            # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
        }


    def forward(self, y: torch.Tensor, c: torch.Tensor):

        """
            Algorithm (1) for jointly training a diffusion model with classifier-free guidance
	    
	    https://arxiv.org/abs/2207.12598
	    
            Arguments
            ----------
            y : torch.Tensor [N, C, H, W]
            c : torch.Tensor [N, n_classes]

            Returns
            ----------
            loss between the predicted noise and the noise \epsilon used to perturb the image
        """

        # --- YOUR CODE HERE ---#
        # TODO
        # 5. dropout class label with probability p_unconditioned
        # mask out class label if class_mask == 0
        class_mask = torch.bernoulli(torch.zeros(c.shape[0]) + (1 - self.drop_prob)).to(self.device)
        c = c * class_mask.unsqueeze(-1)
        # --- YOUR CODE HERE ---#
        # TODO

        return loss


    def sample(self, n_sample: int = 4, size : tuple = None, n_classes : int = 10, g_weight : float = 0.):

        """
            Algorithm (2) for conditional sampling with classifier-free diffusion guidance
            to make the sampling efficient, we concatenate the conditional (c!=0) and unconditional samples (c=0)
            
            https://arxiv.org/abs/2207.12598

            Arguments
            ----------
            n_sample  : int
                number of samples for each class (default: 4)
            size      : tuple (channels, height, height)
                size of the input image (default None)
            n_classes : int
                number of classes/conditions (default is 10)
            g_weight  : float
                guidance weights for conditional generation (default: 0.)

            Returns
            ----------
            y_i       : torch.Tensor [N, C, H, W]
            y_i_steps : torch.Tensor [T, N, C, H, W]

        """
        # 1. sample initial noise y_T ~ N(0, I)
        y_i = torch.randn(n_sample * n_classes, *size, device=self.device)

        # create class label for all classes
        c_i = torch.arange(0, n_classes).to(self.device)
        c_i = c_i.repeat(n_sample)
        c_i = c_i.repeat(2)
        # don't drop class label at test time
        class_mask = torch.ones_like(c_i, device=self.device)
        # double the batch
        # makes second half of batch class label free
        class_mask[n_sample*n_classes:] = 0.
        # convert class label into one hot embedding
        c_i = nn.functional.one_hot(c_i, num_classes=n_classes).type(torch.float)
        # mask out class label if class class_mask == 0
        c_i = c_i * class_mask.unsqueeze(-1)

        # store generated image at different t steps for visualization
        y_i_steps = []

        # 2. do backward process for denoising t = T, ..., 1
        for t in range(self.n_timesteps, 0, -1):

            t_is = torch.tensor([t / self.n_timesteps], device=self.device)
            t_is = t_is.repeat(n_sample*n_classes, 1, 1, 1)

            # double batch to have conditional and unconditional generation
            y_i = y_i.repeat(2, 1, 1, 1)
            t_is = t_is.repeat(2, 1, 1, 1)

            # 3. form the classifier-free guidance score
            eps = self.nn_model(y_i, t_is, c_i)
            # score for conditional generation
            eps1 = eps[:n_sample*n_classes]
            # score for unconditional generation
            eps2 = eps[n_sample*n_classes:]
            # Equation (5)

            # --- YOUR CODE HERE ---#
            # TODO

            # 4. sample extra noise z ~ N(0, I)
            if t > 1:
                z = torch.randn(n_sample*n_classes, *size, device=self.device)
            else:
                z = 0

            # y_i is actually doubled, so take just the first half
            y_i = y_i[:n_sample*n_classes]

            # 5. Equation (4)
            # call it y_i. Here y_i == y_{t-1}
            # --- YOUR CODE HERE ---#
            # TODO
            # y_i =

            # store intermediate results every 20 steps and for last step
            if t % 20 == 0 or t == self.n_timesteps or t < 8:
                y_i_steps.append(y_i.detach().cpu().numpy())

        y_i_steps = np.array(y_i_steps)

        return y_i, y_i_steps


if __name__ == '__main__':

    ddpm = DDPM(nn.Module())






