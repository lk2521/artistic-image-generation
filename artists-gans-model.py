#Best Artworks of All Time - Generative Adversarial Networks (GANs) Project
#### Collection of Paintings of the 50 Most Influential Artists of All Time

####About Dataset:

'''This dataset contains three files:

* `artists.csv`: dataset of information for each artist
* `images.zip`: collection of images (full size), divided in folders and sequentially numbered
* `resized.zip`: same collection but images have been resized and extracted from folder structure.
Using `resized.zip` allows you to download less data and process faster your model.'''

##Download and Explore the Dataset

Using `opendatasets` and download the dataset from [Best Artworks of All Time](https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time)
"""

!pip install opendatasets --upgrade --quiet

import opendatasets as od

dataset_url = 'https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time'

od.download(dataset_url)

import os

data_dir = './best-artworks-of-all-time/resized'

os.listdir(data_dir + '/resized')[:10]

len(os.listdir(data_dir + '/resized'))

"""### Import the dataset into PyTorch



"""

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as T

image_size = 64
batch_size = 128
stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

train_ds = ImageFolder(data_dir, transform=T.Compose([T.Resize(image_size),
                                                     T.CenterCrop(image_size),
                                                     T.ToTensor(),
                                                     T.Normalize(*stats)]))
train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=2, pin_memory=True)

"""Let's create helper functions to denormalize the image tensors and display some sample images from a training batch"""

# Commented out IPython magic to ensure Python compatibility.
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
# %matplotlib inline

def denorm(img_tensors):
    return img_tensors*stats[1][0] + stats[0][0]

def show_images(images, nmax=64):
    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(make_grid(denorm(images.detach()[:nmax]), nrow=8).permute(1, 2, 0))

def show_batch(dl, nmax=64):
    for images, _ in dl:
        show_images(images, nmax)
        break

show_batch(train_dl)

"""##Using a GPU"""

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to( device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

device = get_default_device()
device

train_dl = DeviceDataLoader(train_dl, device)

"""## Discriminator Network

The discriminator takes an image as input, and tries to classify it as "real" or "generated"
"""

import torch.nn as nn

from torch.nn.modules.activation import LeakyReLU
from torch.nn.modules.batchnorm import BatchNorm2d

discriminator = nn.Sequential(
    # in: 3 x 64 x 64

    nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.LeakyReLU(0.2, inplace=True),
    # out: 64 x  32 x 32

    nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.LeakyReLU(0.2, inplace=True),
    # out: 128 x 16 x 16

    nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.LeakyReLU(0.2, inplace=True),
    # out: 256 x 8 x 8

    nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(512),
    nn.LeakyReLU(0.2, inplace=True),
    # out: 512 x 4 x 4

    nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
    # out: 1 x 1 x 1

    nn.Flatten(),
    nn.Sigmoid())

"""Let's move the discriminator to the chosen device"""

discriminator = to_device(discriminator, device)

"""##Generator Network

The input to the generator is typically a vector or a matrix of random numbers (referred to as a latent tensor) which is used as a seed for generating an image. The generator will convert a latent tensor of shape (128, 1, 1) into an image tensor of shape 3 x 64 x 64. To achive this, we'll use the `ConvTranspose2d` layer from PyTorch, which is performs to as a transposed convolution (also referred to as a deconvolution).
"""

latent_size = 128

generator = nn.Sequential(
    # in: latent_size x 1 x 1

    nn.ConvTranspose2d(latent_size, 512, kernel_size=4, stride=1, padding=0, bias=False),
    nn.BatchNorm2d(512),
    nn.ReLU(True),
    # out: 512 x 4 x 4

    nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.ReLU(True),
    # out: 256 x 8 x 8

    nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(True),
    # out: 128 x 16 x 16

    nn.ConvTranspose2d(128, 64,  kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(True),
    # out: 256 x 8 x 8

    nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
    nn.Tanh()
    # out: 3 x 64 x 64
    )

"""We used Tanh activation function for output layer of generator, bacause it always lie in range of `[-1, 1]`

Let's generate some outputs using the generator and view them as images by transforming and denormalizing the output.
"""

xb = torch.randn(batch_size, latent_size, 1, 1) # random latent tensors
fake_images = generator(xb)
print(fake_images.shape)
show_images(fake_images)

"""Let's move the generator to the chosen device"""

generator = to_device(generator, device)

"""## Discriminator Training"""

import torch.nn.functional as F

def train_discriminator(real_images, opt_d):
    # Clear discriminator gradients
    opt_d.zero_grad()

    # Pass real images through discriminator
    real_preds = discriminator(real_images)
    real_targets = torch.ones(real_images.size(0), 1, device=device)
    real_loss = F.binary_cross_entropy(real_preds, real_targets)
    real_score = torch.mean(real_preds).item()

    # Generate fake images
    latent = torch.randn(batch_size, latent_size, 1, 1, device=device)
    fake_images = generator(latent)

    # Pass fake images through discriminator
    fake_targets = torch.zeros(fake_images.size(0), 1, device=device)
    fake_preds = discriminator(fake_images)
    fake_loss = F.binary_cross_entropy(fake_preds, fake_targets)
    fake_score = torch.mean(fake_preds).item()

    # Update discriminator weights
    loss = real_loss + fake_loss
    loss.backward()
    opt_d.step()
    return loss.item(), real_score, fake_score

"""It's important to note that we don't change the weights of the generator model while training the discriminator (opt_d only affects the `discriminator.parameters()`)

## Generator Training
"""

def train_generator(opt_g):
    # Clear generator gradients
    opt_g.zero_grad()

    # Generate fake images
    latent = torch.randn(batch_size, latent_size, 1, 1, device=device)
    fake_images = generator(latent)

    # Try to fool the discriminator
    preds = discriminator(fake_images)
    targets = torch.ones(fake_images.size(0), 1, device=device)
    loss =  F.binary_cross_entropy(preds, targets)

    # Update generator weights
    loss.backward()
    opt_g.step()

    return loss.item()

"""Let's create a directory where we can save intermediate outputs from the generator to visually inspect the progress of the model. We'll also create a helper function to export the generated images."""

from torchvision.utils import save_image

sample_dir = 'generated'
os.makedirs(sample_dir, exist_ok=True)

def save_samples(index, latent_tensors, show=True):
    fake_images = generator(latent_tensors)
    fake_fname = 'generated-images-{0:0=4d}.png'.format(index)
    save_image(denorm(fake_images), os.path.join(sample_dir, fake_fname), nrow=8)
    print('Saving', fake_fname)
    if show:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(fake_images.cpu().detach(), nrow=8).permute(1,2,0))

"""We'll use a fixed set of input vectors to the generator to see how the individual generated images evolve over time as we train the model. Let's save one set of images before we start training our model."""

fixed_latent = torch.randn(64, latent_size, 1, 1, device=device)

save_samples(0, fixed_latent)

"""### Full Training Loop

Let's define a `fit` function to train the discriminator and generator in tandem for each batch of training data. We'll use the Adam optimizer with some custom parameters (betas) that are known to work well for GANs. We will also save some sample generated images at regular intervals for inspection.
"""

from tqdm.notebook import tqdm

def fit(epochs, lr, start_idx=1):
    torch.cuda.empty_cache()

    # Losses and scores
    losses_g = []
    losses_d = []
    real_scores = []
    fake_scores = []

    # Create optimizers
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(epochs):
        for real_images, _ in tqdm(train_dl):
           # Train discriminator
           loss_d, real_score, fake_score = train_discriminator(real_images, opt_d)
           # Train Generator
           loss_g = train_generator(opt_g)

        # Record losses and scores
        losses_g.append(loss_g)
        losses_d.append(loss_d)
        real_scores.append(real_score)
        fake_scores.append(fake_score)

        # Log the losses and scores (last batch)
        print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(
            epoch+1, epochs, loss_g, loss_d, real_score, fake_score))
        # Save generated images
        save_samples(epoch+start_idx, fixed_latent, show=False)

    return losses_g, losses_d, real_scores, fake_scores

"""We are now ready to train the model. Try different learning rates to see if you can maintain the fine balance between the training the generator and the discriminator."""

lr = 0.0002
epochs = 25

history = fit(epochs, lr)

losses_g, losses_d, real_scores, fake_scores = history

"""Now that we have trained the models, we can save checkpoints."""

# Save the model checkpoints
torch.save(generator.state_dict(), 'G.pth')
torch.save(discriminator.state_dict(), 'D.pth')

from IPython.display import Image

Image('./generated/generated-images-0001.png')

Image('./generated/generated-images-0005.png')

Image('./generated/generated-images-0010.png')

Image('./generated/generated-images-0015.png')

Image('./generated/generated-images-0020.png')

Image('./generated/generated-images-0025.png')

history += fit(25, 0.001)

Image('./generated/generated-images-0001.png')

Image('./generated/generated-images-0005.png')

Image('./generated/generated-images-0010.png')

Image('./generated/generated-images-0015.png')

Image('./generated/generated-images-0025.png')

history += fit(10, 0.0005)

Image('./generated/generated-images-0001.png')

Image('./generated/generated-images-0005.png')

Image('./generated/generated-images-0010.png')

history += fit(10, 0.0005)

Image('./generated/generated-images-0001.png')

Image('./generated/generated-images-0005.png')

Image('./generated/generated-images-0010.png')

history += fit(20, 0.00005)

Image('./generated/generated-images-0001.png')

Image('./generated/generated-images-0005.png')

Image('./generated/generated-images-0010.png')

Image('./generated/generated-images-0015.png')

Image('./generated/generated-images-0020.png')

show_batch(DeviceDataLoader(train_dl, 'cpu'))

"""We can also visualize how the loss changes over time. Visualizing losses is quite useful for debugging the training process. For GANs, we expect the generator's loss to reduce over time, without the discriminator's loss getting too high."""

plt.plot(losses_d, '-')
plt.plot(losses_g, '-')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['Discriminator', 'Generator'])
plt.title('Losses');

plt.plot(real_scores, '-')
plt.plot(fake_scores, '-')
plt.xlabel('epoch')
plt.ylabel('score')
plt.legend(['Real', 'Fake'])
plt.title('Scores');

!pip install jovian --upgrade --q

import jovian

# Save the model checkpoints
torch.save(generator.state_dict(), 'G.pth')
torch.save(discriminator.state_dict(), 'D.pth')

project_name = 'deep-learning-GANs-project-artist-paintings'

jovian.commit(project=project_name,
              outputs=['G.pth', 'D.pth'],
              environment=None)

