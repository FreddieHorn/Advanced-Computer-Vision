import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(ConvBlock, self).__init__()
        """
            Convolutional Block with a residual connection
            
            Parameters
            ----------
            in_channels  : int
                number of input channels
            out_channels : int
                number of output channels
        """

        #--- YOUR CODE HERE ---#
        # TODO

    def forward(self, x: torch.Tensor):
        """
            x : tensor [N, C, H, W]
        """

        # --- YOUR CODE HERE ---#
        # TODO
        return out


class Encoder_Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(Encoder_Block, self).__init__()
        """
            Encoder Block in UNet. It consists of two Residual Convolutional Blocks followed by a Maxpooling layer.

            Parameters
            ----------
            in_channels  : int
                number of input channels
            out_channels : int
                number of output channels
        """

    def forward(self, x: torch.Tensor):
        """
            x : tensor [N, C, H, W]
        """
        # --- YOUR CODE HERE ---#
        # TODO

        return x


class Decoder_Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, transpose: int):
        super(Decoder_Block, self).__init__()
        """
            Decoder Block in UNet. It consists of one ConvTransposed layer followed by two Residual Convolutional Blocks.

            Parameters
            ----------
            in_channels  : int
                number of input channels
            out_channels : int
                number of output channels
            transpose    : int
                kernel size and stride for upsampling    
        """

        # --- YOUR CODE HERE ---#
        # TODO

    def forward(self, x: torch.Tensor, x_skip: torch.Tensor):
        """
            x      : tensor from previous decoder or bottleneck layer [N, C, H, W]
            x_skip : tensor from corresponding encoder layer via a skip connection [N, C, H, W]
        """

        # --- YOUR CODE HERE ---#
        # TODO

        return x


class Embedding_Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation: str):
        super(Embedding_Block, self).__init__()
        """
            Embedding Block to provide a guidance to the model. It consists of two fully connected layers 
            with an activation in between.

            Parameters
            ----------
            in_channels  : int
                number of input embedding dimensionality
            out_channels : int
                number of output embedding dimensionality
            activation : str
                activation, could be sin or GELU
        """

        # --- YOUR CODE HERE ---#
        # TODO

    def forward(self, x: torch.Tensor):
        """
            x : tensor for conditions. It could be tensor c [N, n_classes] or t [N, 1, 1, 1]
            please note that: t should be normalized by n_timesteps (t/n_timesteps)
        """

        # flatten the input tensor
        x = x.view(-1, self.in_channels)
        # --- YOUR CODE HERE ---#
        # TODO

        return x


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, n_features: int = 32, n_classes: int = 10, height: int = 32):
        super(UNet, self).__init__()
        """
            UNet Network with guidance layers.

            Parameters
            ----------
            in_channels : int
                number of input channels (default is 3)
            n_features  : int
                number of intermediate features (default is 32)
            n_classes   : int
                number of classes/conditions (default is 10)  
            height      : int
                height of the input image, assuming height == width and dividable by 2 (default is 32)
        """

        self.in_channels = in_channels
        self.n_features = n_features
        self.n_classes = n_classes
        self.height = height

        # Initialize the initial residual convolutional layer
        self.init_conv = ConvBlock(in_channels, n_features)

        # Initialize the encoder layers of the UNet with two levels
        self.encoder_block1 = Encoder_Block(n_features, n_features)
        self.encoder_block2 = Encoder_Block(n_features, 2 * n_features)

        self.to_hidden_vec = nn.Sequential(nn.AvgPool2d(self.height // 4), nn.GELU())

        # Embed the timestep and one-hot class labels with a one-layer fully connected neural network
        self.time_embedding1 = Embedding_Block(1, 2 * n_features, 'sin')
        self.time_embedding2 = Embedding_Block(1, 1 * n_features, 'sin')
        self.class_embedding1 = Embedding_Block(n_classes, 2 * n_features, 'GELU')
        self.class_embedding2 = Embedding_Block(n_classes, 1 * n_features, 'GELU')

        self.decoder_block0 = Decoder_Block(2 * n_features, 2 * n_features, self.height // 4)
        self.decoder_block1 = Decoder_Block(4 * n_features, n_features, 2)
        self.decoder_block2 = Decoder_Block(2 * n_features, n_features, 2)

        # Initialize the final convolutional layers to map to the same number of channels as the input image
        # --- YOUR CODE HERE ---#
        # TODO

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor):
        """
            x : input tensor [N, in_channels, H, W]
            t : timestep [N, 1, 1, 1]. t should be given as a normalized input by n_timesteps (e.g., t/n_timesteps)
            c : label/class condition [N, n_classes]
        """

        # --- YOUR CODE HERE ---#
        # TODO

        return out


if __name__ == '__main__':

    print()
    # check the implementation
    # --- YOUR CODE HERE ---#
    # TODO


