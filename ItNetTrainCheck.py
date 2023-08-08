from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import os
import config
import AItomotools.CTtools.ct_utils as ct
import tomosipo as ts
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from ts_algorithms import fdk
from ItNetDataLoader import loadData
from ItNet import ItNet

dev = torch.device("cuda:2")


def add_zoom_bubble(axes_image,
                    inset_center=(0.25, 0.25),
                    inset_radius=0.2,
                    roi=(0.44, 0.55),
                    zoom=3,
                    edgecolor="red",
                    linewidth=3,
                    alpha=1.0,
                    **kwargs):
    """Add a zoom bubble to an AxesImage

    All coordinates are in (x,y) form where the lowerleft corner is (0, 0)
    and the topright corner is (1,1).

    :param axes_image: `matplotlib.image.AxesImage`

        A return value of `plt.imshow`

    :param inset_center: `(float, float)`

        The center of the inset bubble.

    :param inset_radius: `float`

        The radius of the inset bubble.

    :param roi: `(float, float)`

        The center of the region of interest.

    :param zoom: `float`

        The zoom factor by which the region of interest is magnified.
        The bubble of the region of interest is `zoom` times smaller
        than the bubble of the inset.

    :returns: None
    :rtype: NoneType

    """
    ax = axes_image.axes
    data = axes_image.get_array()
    roi_radius = inset_radius / zoom

    opts = dict(facecolor="none")
    opts["edgecolor"] = edgecolor
    opts["linewidth"] = linewidth
    opts["alpha"] = alpha

    # Make axis for inset bubble.
    axins = ax.inset_axes(
        [
            inset_center[0] - inset_radius,
            inset_center[1] - inset_radius,
            2 * inset_radius,
            2 * inset_radius
        ],
        transform=ax.transAxes,
    )
    axins.axis('off')

    im_inset = axins.imshow(
        data,
        cmap=axes_image.get_cmap(),
        norm=axes_image.norm,
        aspect="auto",            # unknown..
        interpolation="nearest",
        alpha=1.0,
        #vmin=axes_image.get_clim()[0],
        #vmax=axes_image.get_clim()[1],
        origin=axes_image.origin,
        extent=axes_image.get_extent(),
        filternorm=axes_image.get_filternorm(),
        filterrad=axes_image.get_filterrad(),
        # imlim=None,             # imlim is Deprecated
        resample=None,          # No clue..
        url=None,               # No clue..
        data=None,              # This is another way to present args..
    )
    # Show region of interest of the original image
    # This must be in data coordinates.
    axis_to_data = ax.transAxes + ax.transData.inverted()
    lower_left = axis_to_data.transform(np.array(roi) - roi_radius)
    top_right = axis_to_data.transform(np.array(roi) + roi_radius)
    axins.set_xlim(lower_left[0], top_right[0])
    axins.set_ylim(lower_left[1], top_right[1])

    # Clip inset axis to circle and show circle
    patch = patches.Circle(
        inset_center,
        radius=inset_radius,
        transform=ax.transAxes,
        zorder=axins.get_zorder() + 1,               # Show above inset
        **opts,
    )
    im_inset.set_clip_path(patch)
    ax.add_patch(patch)

    # Show bubble at region of interest
    ax.add_patch(
        patches.Circle(
            roi,
            radius=roi_radius,
            transform=ax.transAxes,
            **opts,
        )
    )

    # Draw connection between the two bubbles:
    inset_center = np.array(inset_center)
    roi_center = np.array(roi)
    v = inset_center - roi_center
    d = np.linalg.norm(v)

    ax.add_patch(
        patches.ConnectionPatch(
            # edge of roi bubble
            roi_center + roi_radius / d * v,
            # edge of inset bubble
            roi_center + (d - inset_radius) / d * v,
            'axes fraction', 'axes fraction',
            axesA=ax, axesB=ax, arrowstyle="-",
            **opts
        )
    )

def plots(noisyImg, cleanImg, predImg):
	# initialize our figure

	noPatients = 0
	subDirList = []
	cd = "/local/scratch/public/AItomotools/processed/LIDC-IDRI/"
	for file in os.listdir(cd):
		f = os.path.join(cd, file)
		# checking if it is a file
		if os.path.isfile(f) != True:
			noPatients += 1
			subDirList.append(f)
		
	trainNo = int(np.round(noPatients * 0.8))
	trainList = subDirList[:trainNo]
	testList = subDirList[trainNo:]

	trainDS = loadData(imgPaths=trainList, outputSino=False)
	trainLoader = DataLoader(trainDS, shuffle=False,
	batch_size=config.ITNET_BATCH_SIZE, pin_memory=False)
	
	for (i, (x, y)) in enumerate(trainLoader):
		if i < 1:
			print(i)
			x = x.to(dev)
			y = y.to(dev)
		else:
			break

	noisyImg = x.to(dev)
	noisyImg = noisyImg.cpu().numpy()
	
	figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))

	a = ax[0].imshow(noisyImg[0].T, vmin=0,vmax=2.5)
	b = ax[1].imshow(cleanImg[0].T, vmin=0,vmax=2.5)
	c = ax[2].imshow(predImg[0].T, vmin=0,vmax=2.5)
	# set the titles of the subplots
	ax[0].set_title("Noisy Image")
	ax[1].set_title("Clean Image")
	ax[2].set_title("Predicted Image")
	# set the layout of the figure and display it
	add_zoom_bubble(a)
	add_zoom_bubble(b)
	add_zoom_bubble(c)

	figure.subplots_adjust(right=0.8)
	cbar_ax = figure.add_axes([0.15, 0.3, 0.75, 0.02])
	figure.colorbar(c, cax=cbar_ax, location = "bottom")
	figure.tight_layout()
	ax[0].get_yaxis().set_visible(False)
	ax[0].get_xaxis().set_visible(False)
	ax[1].get_yaxis().set_visible(False)
	ax[1].get_xaxis().set_visible(False)
	ax[2].get_yaxis().set_visible(False)
	ax[2].get_xaxis().set_visible(False)
	
	plt.savefig("/home/obc22/aitomotools/AItomotools/ItNetimgComparison.png")
	

def make_predictions(model, imagePath):
	# set model to evaluation mode
	model.eval()
	# turn off gradient tracking
	
	with torch.no_grad():
		# load the image from disk, swap its color channels, cast it
		# to float data type, and scale its pixel values
		
		noPatients = 0
		subDirList = []
		cd = "/local/scratch/public/AItomotools/processed/LIDC-IDRI/"
		for file in os.listdir(cd):
			f = os.path.join(cd, file)
			# checking if it is a file
			if os.path.isfile(f) != True:
				noPatients += 1
				subDirList.append(f)
			
		trainNo = int(np.round(noPatients * 0.8))
		trainList = subDirList[:trainNo]
		testList = subDirList[trainNo:]

		trainDS = loadData(imgPaths=trainList, outputSino=True)
		trainLoader = DataLoader(trainDS, shuffle=False,
		batch_size=config.ITNET_BATCH_SIZE, pin_memory=False)
		
		for (i, (x, y)) in enumerate(trainLoader):
			if i < 1:
				print(i)
				x = x.to(dev)
				y = y.to(dev)
			else:
				break

		x = x.to(dev)
		y = y.to(dev)

		noisyImg = x
		cleanImg = y
		predImg = model(x).squeeze()
		predImg = predImg.cpu().numpy()
		cleanImg = cleanImg.cpu().numpy()
		noisyImg = noisyImg.cpu().numpy()
		# filter out the weak predictions and convert them to integers
		#predImg = (predImg > config.THRESHOLD) * 255
		#predImg = predImg.astype(np.uint8)
		# prepare a plot for visualization
		plots(noisyImg, cleanImg, predImg)


		return

print("[INFO] loading up test image paths...")
#imagePaths = open(config.TEST_PATHS).read().strip().split("\n")
#imagePaths = np.random.choice(imagePaths, size=10)
imagePaths = ["/local/scratch/public/AItomotools/processed/LIDC-IDRI/LIDC-IDRI-0011/slice_1.npy"]

# load our model from disk and flash it to the current device
print("[INFO] load up model...")
itnet = ItNet().to(dev)
statedict = torch.load("/local/scratch/public/obc22/ItNetTrainCheckpts/AnderTrainedITNETstatedict.pth")
itnet.load_state_dict(statedict)
# iterate over the randomly selected test image paths
for path in imagePaths:
	# make predictions and visualize the results
	make_predictions(itnet, path)
	
loss = torch.load("/local/scratch/public/obc22/ItNetTrainCheckpts/FINALepoch259.pt")['trainLoss']
vloss = torch.load("/local/scratch/public/obc22/ItNetTrainCheckpts/FINALepoch259.pt")['testLoss']

plt.style.use("ggplot")
plt.figure()
plt.plot(loss, label="Training Loss")
plt.plot(vloss, label="Validation Loss")
plt.yscale('log')
plt.title("Training Loss on Dataset")
plt.xlabel("Epoch No.")
plt.ylabel("Loss")
plt.legend(loc="upper right")
plt.savefig("/home/obc22/aitomotools/AItomotools/ItNetLossCheckFINAL.png")

print(np.argmin(vloss))