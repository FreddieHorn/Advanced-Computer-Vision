from tqdm import tqdm
import argparse
import os
import time

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from utils import plot_animation, plot_loss, plot_image, min_max_scale
from networks import Conditional_UNet
from ddpm import DDPM


def parse_args():

    parser = argparse.ArgumentParser('Trainer')
    parser.add_argument('--batch_size', type=int, default=256, help='batch Size [default: 256]')
    parser.add_argument('--n_epochs', type=int, default=20, help='number of epochs [default: 20]')
    parser.add_argument('--log_dir', type=str, default='./log_mnist', help='Log name [default: log_mnist]')
    parser.add_argument('--n_workers', type=int, default=1, help='number of workers [default: 1]')
    parser.add_argument('--l_rate', type=float, default=3e-4, help='learning rate [default: 0.0003]')
    parser.add_argument('--g_weight', type=float, default=[0., 1., 3.],
                        help='guidance weights for conditional generation [default: [0, 1, 3]]')
    parser.add_argument('--n_classes', type=int, default=10, help='number of classes [default: 10]')
    parser.add_argument('--n_features', type=int, default=32,
                        help='number of hidden features/channels in the UNet [default: 32]')
    parser.add_argument('--n_timesteps', type=int, default=400, help='number of timesteps for diffusion [default: 400]')
    parser.add_argument('--n_samples', type=int, default=4,
                        help='samples to be generated for each class during testing [default: 4]')
    parser.add_argument('--im_height', type=int, default=28, help='image height  [default: 28]')
    parser.add_argument('--im_channels', type=int, default=1, help='input channels [default: 1]')

    return parser.parse_args()


def train_mnist(args):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # extract parameters
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    n_timesteps = args.n_timesteps
    n_classes = args.n_classes
    n_features = args.n_features
    l_rate = args.l_rate
    n_samples = args.n_samples
    height = args.im_height
    channels = args.im_channels

    log_dir = args.log_dir
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    g_weight = args.g_weight
    n_workers = args.n_workers

    # define the denoising diffusion probabilistic model
    ddpm = DDPM(nn_model=Conditional_UNet.UNet(in_channels=channels,
                                               n_features=n_features,
                                               n_classes=n_classes,
                                               height=height),
                beta1=1e-4,
                beta2=0.02,
                n_timesteps=n_timesteps,
                device=device,
                drop_prob=0.1).to(device)

    # prepare MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = MNIST("./data",
                    train=True,
                    download=True,
                    transform=transform)

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=n_workers,
                            drop_last=True)

    # define optimizer
    optimizer = torch.optim.Adam(ddpm.parameters(), lr=l_rate)

    # training loop

    for epoch in range(n_epochs):

        # training
        ddpm.train()
        loss_smoothed = 0

        time.sleep(1)

        pbar = tqdm(dataloader, total=len(dataloader), smoothing=0.9, postfix=" training")

        for y, c in pbar:
            # zeroing out gradients
            optimizer.zero_grad()
            # scale MNIST dataset to the range -1, +1
            y = min_max_scale(y, 0, 1, -1, 1)

            y = y.to(device)
            c = c.to(device)

            # convert class label into one hot embedding
            # --- YOUR CODE HERE ---#
            # TODO convert class label to one hot embedding
            # c =

            # compute loss and do backpropagation
            # --- YOUR CODE HERE ---#
            # TODO compute loss and do backpropagation
            # loss =

            if loss_smoothed == 0:
                loss_smoothed = loss.item()
            else:
                loss_smoothed = 0.9 * loss_smoothed + 0.1 * loss.item()

            pbar.set_description("epoch: %s >> smoothed loss: %.4f" % (epoch+1, loss_smoothed))

        # --- YOUR CODE HERE ---#
        # TODO compute mean loss for this epoch and store it

        # save the model
        if epoch % 5 == 0 or epoch == int(n_epochs - 1):
            # --- YOUR CODE HERE ---#
            # TODO save trained model
            print()

        if epoch % 5 == 0 or epoch == int(n_epochs - 1):
            # generate samples for testing
            ddpm.eval()

            # restore class label from one-hot encoding
            c = torch.argmax(c, dim=1)

            with torch.no_grad():

                # generate samples for each guidance weight
                for g_weight_i in g_weight:

                    y_sampled, y_sampled_steps = ddpm.sample(n_samples, (channels, height, height), n_classes, g_weight=g_weight_i)
                    # for testing, we use the images of currently generated samples (top rows)
                    # followed by real images (bottom rows)
                    y_real = torch.Tensor(y_sampled.shape).to(device)

                    for k in range(n_classes):
                        for j in range(n_samples):
                            try:
                                idx = torch.squeeze((c == k).nonzero())[j]
                            except:
                                idx = 0

                            y_real[k+(j*n_classes)] = y[idx]

                    y_all = torch.cat([y_sampled, y_real], dim=0)

                    # denormalize MNIST
                    y_all = min_max_scale(y_all, -1, 1, 0, 1)

                    # save images
                    save_path = os.path.join(log_dir,
                                             'generated_images_epoch{}_guidance_weight{}.png'.format(epoch + 1,
                                                                                                     g_weight_i))
                    plot_image(y_all, n_classes, save_path)

                    # save animated images
                    plot_animation(y_sampled_steps, n_samples, n_classes, log_dir, epoch, g_weight_i)

            print()

    # plot the loss
    # --- YOUR CODE HERE ---#
    # TODO plot the loss for training and save it
    # you can use the function plot_loss


if __name__ == "__main__":

    args = parse_args()
    train_mnist(args)

