import numpy as np
import torch
import tomosipo as ts
import matplotlib.pyplot as plt
import AItomotools.CTtools.ct_utils as ct
import pathlib

def default():
#%% Demo on how to create a sinogram from an image and simulate projections, for 2D, using tomosipo and AItomotools
# ===================================================================================================================
# by: Ander Biguri

# If you are here is because you want to do tomography with AI.
# Therefore, to start with, we need to learn how to do tomography.
# This demo teaches how to define a tomographic geometry, an image, and produce sinograms (CT measurements) from it.
# It also shows how to simulate realistic CT noise.
# AItomotools has various backends, but we are using tomosipo in this case.


    #%% Create image, process
    # Create a phantom containing a small cube. In your code, this will be your data
    phantom = np.ones((512, 512), dtype="float32") * -1000  # lets assume its 512^2
    phantom[200:250, 200:250] = 300

    # As we want a 2D image but tomosipo deals with 3D data, lets make it 3D with a singleton z dimension
    phantom = np.expand_dims(phantom, 0)


    phantom_mu = ct.from_HU_to_mu(phantom)


    # We can also breakdown the HU image into a segmented tissue ID image. Check function definition for documentatio of tissue idx.

    # Lets use the mu, as this is the real measurement (mu is what would measure, as HUs are a post-processing change of units that doctors like)
    #phantom = phantom_mu

    # CT images can be in different units. The phantom avobe has been defined in Hounsfield Units (HUs)
    # CTtools.ct_utils has a series of functions to transfrom from 3 units: Hounsfield Units (HUs), linear attenuation coefficien (mus) or normalized images

    #%% Create operator

    # We need to define a "virtual CT machine" that simualtes the data. This is, in mathematical terms, an "operator".

    # Define volume geometry. Shape is the shape in elements, and size is in real life units (e.g. mm)
    # Note: we make Z "thick" just because we are going to simulate 2D. for 3D be more careful with this value.
    vg = ts.volume(shape=(1, *phantom.shape[1:]), size=(5, 300, 300))
    # Define acquisition geometry. We want fan beam, so lets make a "cone" beam and make it 2D. We also need sod and sdd, so we set them up to something medically reasonable.
    pg = ts.cone(
        angles=360, shape=(1, 900), size=(1, 900), src_orig_dist=575, src_det_dist=1050
    )
    # A is now an operator.
    A = ts.operator(vg, pg)

    dev = torch.device("cuda")
    phantom = torch.from_numpy(phantom).to(dev)

    # But given we already defined an operator, we can just do (its the same):
    sino = A(phantom)
    # For noise simulation, a good approximation of CT noise is to add Poisson noise to the non-log transformed sinograms,
    # with some gaussian noise to account for the detector electronic noise and detector crosstalk.
    # A typical CT scan in a hospital will have I0=10000 photon counts in air. I0=1000 will produce an severely noisy image.
    # You should be cool with not touching the rest of the parameters.
    sino_noisy = ct.sinogram_add_noise(
        sino, I0=10000, sigma=5, crosstalk=0.05, flat_field=None, dark_field=None
    )



    plt.figure()
    plt.subplot(121)
    plt.imshow(sino[0].T)
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(sino_noisy[0].T)
    plt.colorbar()
    plt.savefig("Sino.png")
    return

def getSino(imgClean):
    #takes clean img and turns into a sinogram
    #vg = ts.volume(shape=(1, *imgClean.shape[1:]), size=(5, 300, 300))
    vg = ts.volume(shape=(1, *imgClean.shape[1:]), size=(300/imgClean.shape[1], 300, 300))
    # Define acquisition geometry. We want fan beam, so lets make a "cone" beam and make it 2D. We also need sod and sdd, so we set them up to something medically reasonable.
    pg = ts.cone(
        angles=360, shape=(1, 900), size=(1, 900), src_orig_dist=575, src_det_dist=1050
    )
    # A is now an operator.
    A = ts.operator(vg, pg)
    return A(imgClean)


def sinoTest():
    imgToLoad = "/local/scratch/public/AItomotools/processed/LIDC-IDRI/LIDC-IDRI-1001/slice_19.npy"
    imgClean = np.load(imgToLoad)
    imgClean = ct.from_HU_to_mu(imgClean)
    imgClean = torch.from_numpy(imgClean)#.to(self.dev) #make in gpu
    imgClean = np.expand_dims(imgClean, 0).astype("float32")


    sino = getSino(imgClean)

    sino_noisy = ct.sinogram_add_noise(
    sino, I0=10000, sigma=5, crosstalk=0.05, flat_field=None, dark_field=None
    )
    sino_noisy = torch.from_numpy(sino_noisy)
    sino_noisy = sino_noisy.type(torch.float32)

    #sino = sino.detach().cpu().numpy()
    #sino_noisy = sino_noisy.detach().cpu().numpy()

    plt.figure()
    plt.subplot(121)
    plt.imshow(sino[0].T)
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(sino_noisy[0].T)
    plt.colorbar()
    plt.savefig("/home/obc22/aitomotools/AItomotools/Sino.png")
    return

sinoTest()