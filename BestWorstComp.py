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
from UNet import UNet
from TVMin import tvMin
from skimage.metrics import structural_similarity as ssim


dev = torch.device("cuda:2")

"""Script to find and generate images of the best and worst reconstructions each method has produced on the givrn data set"""


def add_zoom_bubble(axes_image,
                    inset_center=(0.25, 0.25),
                    inset_radius=0.2,
                    roi=(0.44, 0.55),
                    zoom=3,
                    edgecolor="red",
                    linewidth=3,
                    alpha=1.0,
                    **kwargs):
    """Code taken from https://github.com/ahendriksen/noise2inverse/tree/3841c471130c6b3638363f6dfbaedc214b848131
    
    Add a zoom bubble to an AxesImage

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

def plots(unet, itnet):
	"""Code to plot the figures
 
 	unet, itnet are both input models"""


	vg = ts.volume(shape=(1, *(512,512)), size=(300/512, 300, 300))
	pg = ts.cone(
        angles=360, shape=(1, 900), size=(1, 900), src_orig_dist=575, src_det_dist=1050)
	A = ts.operator(vg, pg)
	
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
	validateNo = int(np.round(noPatients * 0.1))
	testNo = noPatients - trainNo - validateNo
	trainList = subDirList[:trainNo]
	validateList = subDirList[trainNo:trainNo+validateNo]
	testList = subDirList[trainNo+validateNo:]
	trainDS = loadData(imgPaths=testList, outputSino=True)
	trainLoader = DataLoader(trainDS, shuffle=False,
	batch_size=config.ITNET_BATCH_SIZE, pin_memory=False)
	
	uB = 0
	uW = 1
	
	iB = 0
	iW = 1
	
	tB = 0
	tW = 1
	
	unet.eval()
	itnet.eval()
	with torch.no_grad():
		for (i, (x, y)) in enumerate(trainLoader):
			print(i)
			x = x.to(dev)
			y = y.to(dev)
			
			noisyImg = x
			cleanImg = y
			FDKpredImg = torch.zeros_like(y)
			for j in range(x.shape[0]):
				FDKpredImg[j]=fdk(A, x[j])
			TVpredImg = tvMin(x).forward()
			UpredImg = unet(FDKpredImg).squeeze().cpu().numpy()
			IpredImg = itnet(x).squeeze().cpu().numpy()
			FDKpredImg = FDKpredImg.cpu().numpy()
			TVpredImg = TVpredImg.cpu().numpy()
			cleanImg = cleanImg.cpu().numpy()
			noisyImg = noisyImg.cpu().numpy()

			fdkErr = ssim(cleanImg[0][0], FDKpredImg[0][0], data_range = FDKpredImg[0][0].max()-FDKpredImg[0][0].min())
			tvErr = ssim(cleanImg[0][0], TVpredImg[0], data_range = TVpredImg[0].max()-TVpredImg[0].min())
			uErr = ssim(cleanImg[0][0], UpredImg[0], data_range = UpredImg[0].max()-UpredImg[0].min())
			iErr = ssim(cleanImg[0][0], IpredImg[0], data_range = IpredImg[0].max()-IpredImg[0].min())

			if tvErr > iErr:
				print("HEY")
				figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 6))
				a = ax[0].imshow(cleanImg[0].T, vmin=0,vmax=2.5)
				b = ax[1].imshow(TVpredImg[0][0].T, vmin=0,vmax=2.5)
				c = ax[2].imshow(IpredImg[0].T, vmin=0,vmax=2.5)
				ax[0].set_title("Clean Image")
				ax[1].set_title("Predicted Image (TV Min), SSIM = {:.3f}".format(tvErr))
				ax[2].set_title("Predicted Image (ItNet), SSIM = {:.3f}".format(iErr))
				add_zoom_bubble(a)
				add_zoom_bubble(b)
				add_zoom_bubble(c)
				figure.subplots_adjust(right=0.8)
				figure.tight_layout()
				ax[0].get_yaxis().set_visible(False)
				ax[0].get_xaxis().set_visible(False)
				ax[1].get_yaxis().set_visible(False)
				ax[1].get_xaxis().set_visible(False)
				ax[2].get_yaxis().set_visible(False)
				ax[2].get_xaxis().set_visible(False)
				plt.savefig("/home/obc22/aitomotools/AItomotools/IworseThanTV.png")


			if tvErr > tB:
				tB = tvErr
				print("tB")
				figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 6))
				a = ax[0].imshow(cleanImg[0].T, vmin=0,vmax=2.5)
				b = ax[1].imshow(FDKpredImg[0][0].T, vmin=0,vmax=2.5)
				c = ax[2].imshow(TVpredImg[0].T, vmin=0,vmax=2.5)
				ax[0].set_title("Clean Image")
				ax[1].set_title("Predicted Image (FDK), SSIM = {:.3f}".format(fdkErr))
				ax[2].set_title("Predicted Image (Best), SSIM = {:.3f}".format(tvErr))
				add_zoom_bubble(a)
				add_zoom_bubble(b)
				add_zoom_bubble(c)
				figure.subplots_adjust(right=0.8)
				figure.tight_layout()
				ax[0].get_yaxis().set_visible(False)
				ax[0].get_xaxis().set_visible(False)
				ax[1].get_yaxis().set_visible(False)
				ax[1].get_xaxis().set_visible(False)
				ax[2].get_yaxis().set_visible(False)
				ax[2].get_xaxis().set_visible(False)
				plt.savefig("/home/obc22/aitomotools/AItomotools/TVImgB.png")
				
			if tvErr < tW:
				tW = tvErr
				print("tW")
				figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 6))
				a = ax[0].imshow(cleanImg[0].T, vmin=0,vmax=2.5)
				b = ax[1].imshow(FDKpredImg[0][0].T, vmin=0,vmax=2.5)
				c = ax[2].imshow(TVpredImg[0].T, vmin=0,vmax=2.5)
				ax[0].set_title("Clean Image")
				ax[1].set_title("Predicted Image (FDK), SSIM = {:.3f}".format(fdkErr))
				ax[2].set_title("Predicted Image (Worst), SSIM = {:.3f}".format(tvErr))
				add_zoom_bubble(a)
				add_zoom_bubble(b)
				add_zoom_bubble(c)
				figure.subplots_adjust(right=0.8)
				cbar_ax = figure.add_axes([0.15, 0.05, 0.75, 0.02])
				figure.colorbar(c, cax=cbar_ax, location = "bottom")
				figure.tight_layout()
				ax[0].get_yaxis().set_visible(False)
				ax[0].get_xaxis().set_visible(False)
				ax[1].get_yaxis().set_visible(False)
				ax[1].get_xaxis().set_visible(False)
				ax[2].get_yaxis().set_visible(False)
				ax[2].get_xaxis().set_visible(False)
				plt.savefig("/home/obc22/aitomotools/AItomotools/TVImgW.png")

			
			if uErr > uB:
				uB = uErr
				print("uB")
				figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 6))
				a = ax[0].imshow(cleanImg[0].T, vmin=0,vmax=2.5)
				b = ax[1].imshow(FDKpredImg[0][0].T, vmin=0,vmax=2.5)
				c = ax[2].imshow(UpredImg[0].T, vmin=0,vmax=2.5)
				ax[0].set_title("Clean Image")
				ax[1].set_title("Predicted Image (FDK), SSIM = {:.3f}".format(fdkErr))
				ax[2].set_title("Predicted Image (Best), SSIM = {:.3f}".format(uErr))
				add_zoom_bubble(a)
				add_zoom_bubble(b)
				add_zoom_bubble(c)
				figure.subplots_adjust(right=0.8)
				figure.tight_layout()
				ax[0].get_yaxis().set_visible(False)
				ax[0].get_xaxis().set_visible(False)
				ax[1].get_yaxis().set_visible(False)
				ax[1].get_xaxis().set_visible(False)
				ax[2].get_yaxis().set_visible(False)
				ax[2].get_xaxis().set_visible(False)
				plt.savefig("/home/obc22/aitomotools/AItomotools/UImgB.png")
				
			if uErr < uW:
				uW = uErr
				print("uW")
				figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 6))
				a = ax[0].imshow(cleanImg[0].T, vmin=0,vmax=2.5)
				b = ax[1].imshow(FDKpredImg[0][0].T, vmin=0,vmax=2.5)
				c = ax[2].imshow(UpredImg[0].T, vmin=0,vmax=2.5)
				ax[0].set_title("Clean Image")
				ax[1].set_title("Predicted Image (FDK), SSIM = {:.3f}".format(fdkErr))
				ax[2].set_title("Predicted Image (Worst), SSIM = {:.3f}".format(uErr))
				add_zoom_bubble(a)
				add_zoom_bubble(b)
				add_zoom_bubble(c)
				figure.subplots_adjust(right=0.8)
				cbar_ax = figure.add_axes([0.15, 0.05, 0.75, 0.02])
				figure.colorbar(c, cax=cbar_ax, location = "bottom")
				figure.tight_layout()
				ax[0].get_yaxis().set_visible(False)
				ax[0].get_xaxis().set_visible(False)
				ax[1].get_yaxis().set_visible(False)
				ax[1].get_xaxis().set_visible(False)
				ax[2].get_yaxis().set_visible(False)
				ax[2].get_xaxis().set_visible(False)
				plt.savefig("/home/obc22/aitomotools/AItomotools/UImgW.png")
				
			if iErr > iB:
				iB = iErr
				print("iB")
				figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 6))
				a = ax[0].imshow(cleanImg[0].T, vmin=0,vmax=2.5)
				b = ax[1].imshow(FDKpredImg[0][0].T, vmin=0,vmax=2.5)
				c = ax[2].imshow(IpredImg[0].T, vmin=0,vmax=2.5)
				ax[0].set_title("Clean Image")
				ax[1].set_title("Predicted Image (FDK), SSIM = {:.3f}".format(fdkErr))
				ax[2].set_title("Predicted Image (Best), SSIM = {:.3f}".format(iErr))
				add_zoom_bubble(a)
				add_zoom_bubble(b)
				add_zoom_bubble(c)
				figure.subplots_adjust(right=0.8)
				figure.tight_layout()
				ax[0].get_yaxis().set_visible(False)
				ax[0].get_xaxis().set_visible(False)
				ax[1].get_yaxis().set_visible(False)
				ax[1].get_xaxis().set_visible(False)
				ax[2].get_yaxis().set_visible(False)
				ax[2].get_xaxis().set_visible(False)
				plt.savefig("/home/obc22/aitomotools/AItomotools/IImgB.png")
				
			if iErr < iW:
				iW = iErr
				print("iW")
				figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 6))
				a = ax[0].imshow(cleanImg[0].T, vmin=0,vmax=2.5)
				b = ax[1].imshow(FDKpredImg[0][0].T, vmin=0,vmax=2.5)
				c = ax[2].imshow(IpredImg[0].T, vmin=0,vmax=2.5)
				ax[0].set_title("Clean Image")
				ax[1].set_title("Predicted Image (FDK), SSIM = {:.3f}".format(fdkErr))
				ax[2].set_title("Predicted Image (Worst), SSIM = {:.3f}".format(iErr))
				add_zoom_bubble(a)
				add_zoom_bubble(b)
				add_zoom_bubble(c)
				figure.subplots_adjust(right=0.8)
				cbar_ax = figure.add_axes([0.15, 0.05, 0.75, 0.02])
				figure.colorbar(c, cax=cbar_ax, location = "bottom")
				figure.tight_layout()
				ax[0].get_yaxis().set_visible(False)
				ax[0].get_xaxis().set_visible(False)
				ax[1].get_yaxis().set_visible(False)
				ax[1].get_xaxis().set_visible(False)
				ax[2].get_yaxis().set_visible(False)
				ax[2].get_xaxis().set_visible(False)
				plt.savefig("/home/obc22/aitomotools/AItomotools/IImgW.png")
				
				figure, ax = plt.subplots(nrows=2, ncols=3, figsize=(12, 10))
				#a = ax[0,0].imshow(noisyImg[0].T, vmin=0,vmax=2.5)
				b = ax[0,1].imshow(cleanImg[0][0].T, vmin=0,vmax=2.5)
				c = ax[0,2].imshow(FDKpredImg[0][0].T, vmin=0,vmax=2.5)
				# set the titles of the subplots
				#ax[0,0].set_title("Noisy Image")
				ax[0,1].set_title("Clean Image")
				ax[0,2].set_title("Predicted Image (FDK), SSIM = {:.3f}".format(fdkErr))
				ax[1,0].set_title("Predicted Image (TV Min), SSIM = {:.3f}".format(tvErr))
				ax[1,1].set_title("Predicted Image (UNet), SSIM = {:.3f}".format(uErr))
				ax[1,2].set_title("Predicted Image (ItNet), SSIM = {:.3f}".format(iErr))
				# set the layout of the figure and display it
				#add_zoom_bubble(a)
				add_zoom_bubble(b)
				add_zoom_bubble(c)
				d = ax[1,0].imshow(TVpredImg[0].T, vmin=0,vmax=2.5)
				e = ax[1,1].imshow(UpredImg[0].T, vmin=0,vmax=2.5)
				f = ax[1,2].imshow(IpredImg[0].T, vmin=0,vmax=2.5)
				# set the titles of the subplots
				add_zoom_bubble(d)
				add_zoom_bubble(e)
				add_zoom_bubble(f)

				figure.subplots_adjust(right=0.8)
				cbar_ax = figure.add_axes([0.15, 0.05, 0.75, 0.02])
				figure.colorbar(c, cax=cbar_ax, location = "bottom")
				figure.tight_layout()
				ax[0,0].get_yaxis().set_visible(False)
				ax[0,0].get_xaxis().set_visible(False)
				ax[0,1].get_yaxis().set_visible(False)
				ax[0,1].get_xaxis().set_visible(False)
				ax[0,2].get_yaxis().set_visible(False)
				ax[0,2].get_xaxis().set_visible(False)
				ax[1,0].get_yaxis().set_visible(False)
				ax[1,0].get_xaxis().set_visible(False)
				ax[1,1].get_yaxis().set_visible(False)
				ax[1,1].get_xaxis().set_visible(False)
				ax[1,2].get_yaxis().set_visible(False)
				ax[1,2].get_xaxis().set_visible(False)
				
				plt.savefig("/home/obc22/aitomotools/AItomotools/IWorstVOthers.png")
	return

print("[INFO] load up model...")
itnet = ItNet().to(dev)
itnetSD = torch.load("/local/scratch/public/obc22/ItNetTrainCheckpts/FINALepoch209.pt")['model_state_dict']
itnet.load_state_dict(itnetSD)

unet = UNet().to(dev)
unetSD = torch.load("/local/scratch/public/obc22/UNetTrainCheckpts/trainedUNETstatedict.pth")
unet.load_state_dict(unetSD)
# iterate over the randomly selected test image paths
plots(unet, itnet)
	
