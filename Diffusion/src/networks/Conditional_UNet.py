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
        self.subblock1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.subblock2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

        # If same channel dimension, we'll use the residual connection from the original input.
        # If not, then we'll have a residual connection from the first subblock.
        self.in_out_channels_equal = in_channels == out_channels



    def forward(self, x: torch.Tensor):
        out1 = self.subblock1(x)
        out = self.subblock2(out1)

        if self.in_out_channels_equal:
            out += x
        else:
            out += out1
        
        out /= torch.sqrt(torch.tensor(2)) # Normalize by sqrt(2) (Why?)
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
        self.sequential = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=out_channels),
            ConvBlock(in_channels=out_channels, out_channels=out_channels),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x: torch.Tensor):
        """
            x : tensor [N, C, H, W]
        """
        return self.sequential(x)


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

        self.sequential = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=transpose, stride=transpose),
            ConvBlock(in_channels=out_channels, out_channels=out_channels),
            ConvBlock(in_channels=out_channels, out_channels=out_channels)
        )


    def forward(self, x: torch.Tensor, x_skip: torch.Tensor):
        """
            x      : tensor from previous decoder or bottleneck layer [N, C, H, W]
            x_skip : tensor from corresponding encoder layer via a skip connection [N, C, H, W]
        """

        if x_skip != None:
            x = torch.cat([x, x_skip], dim=1)
        out = self.sequential(x)
    
        return out


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
        self.in_channels = in_channels

        self.linear1 = nn.Linear(in_features=in_channels, out_features=out_channels)
        if activation == "sin":
            self.activation = torch.sin
        elif activation == "GELU":
            self.activation = nn.GELU()
        else:
            print("Error. Wrong embedding activation type given")

        self.linear2 = nn.Linear(in_features=out_channels, out_features=out_channels)


    def forward(self, x: torch.Tensor):
        """
            x : tensor for conditions. It could be tensor c [N, n_classes] or t [N, 1, 1, 1]
            please note that: t should be normalized by n_timesteps (t/n_timesteps)
        """

        # flatten the input tensor
        x = x.view(-1, self.in_channels)
        out = self.linear1(x)
        out = self.activation(out)
        out = self.linear2(out)
        out = out.squeeze()
        return out


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
        self.final_conv = nn.Sequential( #  I don't know how to set it exactly, so I just try this:
            nn.Conv2d(2 * n_features, n_features, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, n_features),
            nn.Conv2d(n_features, in_channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor):
        """
            x : input tensor [N, in_channels, H, W]
            t : timestep [N, 1, 1, 1]. t should be given as a normalized input by n_timesteps (e.g., t/n_timesteps)
            c : label/class condition [N, n_classes]
        """
        out_init = self.init_conv(x)
        out_enc1 = self.encoder_block1(out_init)
        out_enc2 = self.encoder_block2(out_enc1)

        out_hid = self.to_hidden_vec(out_enc2)
        out_dec0 = self.decoder_block0(out_hid, None)

        out_time1 = self.time_embedding1(t)
        out_class1 = self.class_embedding1(c)

        out_dec0 *= out_class1.unsqueeze(0).unsqueeze(2).unsqueeze(3) # unsqueezing for dimensional match
        out_dec0 += out_time1.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        out_dec1 = self.decoder_block1(out_dec0, out_enc2)

        out_time2 = self.time_embedding2(t)
        out_class2 = self.class_embedding2(c)

        out_dec1 *= out_class2.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        out_dec1 += out_time2.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        out_dec2 = self.decoder_block2(out_dec1, out_enc1)

        out_final = self.final_conv(torch.cat([out_init, out_dec2], dim=1))


        return out_final


if __name__ == '__main__':

    x = torch.randn((1, 3, 32, 32))
    t = torch.randn((1, 1, 1, 1))
    c = torch.randn((1, 10))


    # Test for both GPU (if available) and cpu

    devices = [torch.device("cpu")]

    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    else:
        print("no cuda available")

    for device in devices:
        x = x.to(device)
        t = t.to(device)
        c = c.to(device)

        model = UNet(in_channels=3, n_features=32, n_classes=10, height=32)

        model = model.to(device)

        test_result = model(x, t, c)

        num_params = sum(p.numel() for p in model.parameters())
        print("Number of parameters: ", num_params)

    



