import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
import os
from torchvision.utils import save_image, make_grid

def min_max_scale(array, min_array, max_array, min_new=-1, max_new=1):
    array = ((max_new - min_new) * (array - min_array) / (max_array - min_array)) + min_new
    return array


def unorm(x):
    """
        brings an image to unity norm >> results in an image in the range of [0, 1]

        Parameters
        ----------
        x : numpy array representing an image [C, H, W]
    """

    xmax = x.max((1, 2))
    xmin = x.min((1, 2))
    norm_x = (x - xmin[:, None, None])/(xmax[:, None, None] - xmin[:, None, None])

    return norm_x


def norm_all(x_steps, n_t, n_s):
    """
        brings images to unity norm >> results in images in the range of [0,1]

        Parameters
        ----------
        x_steps : numpy array representing images (T,S,C,H,W)
        n_t     : number of sampled timesteps
        n_s     : number of samples
    """

    n_x_steps = np.zeros_like(x_steps)
    for t in range(n_t):
        for s in range(n_s):
            n_x_steps[t, s] = unorm(x_steps[t, s])
    return n_x_steps


def plot_animation(y_sampled_steps, n_samples, n_classes, log_dir, epoch, g_weight, rgb=False):
    # create gif of images evolving over time, based on y_sampled_steps
    fig, axs = plt.subplots(nrows=n_samples, ncols=n_classes, sharex=True, sharey=True, figsize=(8, 3))

    def animate_diff(i, y_sampled_steps):
        #print(f'gif animating frame {i} of {y_sampled_steps.shape[0]}', end='\r')
        plots = []
        for row in range(int(n_samples)):
            for col in range(n_classes):
                axs[row, col].clear()
                axs[row, col].set_xticks([])
                axs[row, col].set_yticks([])
                # plots.append(axs[row, col].imshow(y_sampled_steps[i,(row*n_classes)+col,0],cmap='gray'))
                if rgb:
                    plots.append(axs[row, col].imshow(np.moveaxis(y_sampled_steps[i, (row * n_classes) + col, :], 0, -1)))
                else:
                    plots.append(axs[row, col].imshow(y_sampled_steps[i, (row *n_classes) + col, 0],
                                                      cmap='Greys'))
                                                      #vmin=(-y_sampled_steps[i]).min(), vmax=(-y_sampled_steps[i]).max()))
        return plots

    y_sampled_steps = norm_all(y_sampled_steps, y_sampled_steps.shape[0], n_samples * n_classes)

    ani = FuncAnimation(fig, animate_diff, fargs=[y_sampled_steps],
                        interval=200, blit=False, repeat=True, frames=y_sampled_steps.shape[0])
    save_path = os.path.join(log_dir, 'generated_gif_epoch{}_guidance_weight{}.gif'.format(epoch + 1, g_weight))
    ani.save(save_path, dpi=100, writer=PillowWriter(fps=5))
    print('saved gif to %s' % save_path)
    plt.close()


def plot_loss(mean_loss_train, save_path):

    # Create count of the number of epochs
    range_epochs = range(1, len(mean_loss_train) + 1)
    # Visualize loss history
    plt.plot(range_epochs, mean_loss_train, 'r-')
    plt.legend(['Training Loss'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(np.arange(0, len(mean_loss_train), len(mean_loss_train)//10))
    plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_image(y_all, n_classes, save_path, inverse=True):

    if inverse:
        grid = make_grid(y_all * -1 + 1, nrow=n_classes)
    else:
        grid = make_grid(y_all, nrow=n_classes)

    save_image(grid, save_path)
    print('saved image to %s' % save_path)
    plt.close()
