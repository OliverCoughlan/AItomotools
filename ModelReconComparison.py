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

def plots(noisyImg, cleanImg, FDKpredImg,
	  TVpredImg,UpredImg,IpredImg, plotfName):
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
		

	
	"""for (i, (x, y)) in enumerate(trainLoader):
		if i < 1:
			print(i)
			x = x.to(dev)
			y = y.to(dev)
		else:
			break"""





	fdkErr = ssim(cleanImg[0][0], FDKpredImg[0][0], data_range = FDKpredImg[0][0].max()-FDKpredImg[0][0].min())
	tvErr = ssim(cleanImg[0][0], TVpredImg[0], data_range = TVpredImg[0].max()-TVpredImg[0].min())
	uErr = ssim(cleanImg[0][0], UpredImg[0], data_range = UpredImg[0].max()-UpredImg[0].min())
	iErr = ssim(cleanImg[0][0], IpredImg[0], data_range = IpredImg[0].max()-IpredImg[0].min())

	figure, ax = plt.subplots(nrows=2, ncols=3, figsize=(12, 10))

	#a = ax[0,0].imshow(noisyImg[0].T, vmin=0,vmax=2.5)
	b = ax[0,0].imshow(cleanImg[0][0].T, vmin=0,vmax=2.5)
	c = ax[0,2].imshow(FDKpredImg[0][0].T, vmin=0,vmax=2.5)
	# set the titles of the subplots
	#ax[0,0].set_title("Noisy Image")
	ax[0,0].set_title("Clean Image")
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
	figure.delaxes(ax[0,1])
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
	
	plt.savefig("/home/obc22/aitomotools/AItomotools/ImgComparisons/{}.png".format(plotfName))
	

def make_predictions(unet, itnet, imagePath):
	# set model to evaluation mode
	unet.eval()
	itnet.eval()
	# turn off gradient tracking
	vg = ts.volume(shape=(1, *(512,512)), size=(300/512, 300, 300))
	pg = ts.cone(
        angles=360, shape=(1, 900), size=(1, 900), src_orig_dist=575, src_det_dist=1050)
	A = ts.operator(vg, pg)

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
		validateNo = int(np.round(noPatients * 0.1))
		testNo = noPatients - trainNo - validateNo

		trainList = subDirList[:trainNo]
		validateList = subDirList[trainNo:trainNo+validateNo]
		testList = subDirList[trainNo+validateNo:]

		trainDS = loadData(imgPaths=testList, outputSino=True)
		trainLoader = DataLoader(trainDS, shuffle=False,
		batch_size=config.ITNET_BATCH_SIZE, pin_memory=False)
		
		'''for (i, (x, y,z)) in enumerate(trainLoader):
			if i < 1:
				print(i)
				x = x.to(dev)
				y = y.to(dev)
			else:
				break'''
		for (i, (x, y,z)) in enumerate(trainLoader):
			x = x.to(dev)
			y = y.to(dev)
			plotfName = (z[0][-21:-4].replace('/','-'))
			print(i)

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
			# filter out the weak predictions and convert them to integers
			#predImg = (predImg > config.THRESHOLD) * 255
			#predImg = predImg.astype(np.uint8)
			# prepare a plot for visualization
			plots(noisyImg, cleanImg, FDKpredImg,
			TVpredImg, UpredImg, IpredImg, plotfName)


		return

print("[INFO] loading up test image paths...")
#imagePaths = open(config.TEST_PATHS).read().strip().split("\n")
#imagePaths = np.random.choice(imagePaths, size=10)
imagePaths = ["/local/scratch/public/AItomotools/processed/LIDC-IDRI/LIDC-IDRI-0001/slice_0.npy"]

# load our model from disk and flash it to the current device
print("[INFO] load up model...")
itnet = ItNet().to(dev)
itnetSD = torch.load("/local/scratch/public/obc22/ItNetTrainCheckpts/FINALepoch259.pt")['model_state_dict']
itnet.load_state_dict(itnetSD)

unet = UNet().to(dev)
unetSD = torch.load("/local/scratch/public/obc22/UNetTrainCheckpts/trainedUNETstatedict.pth")
unet.load_state_dict(unetSD)
# iterate over the randomly selected test image paths
for path in imagePaths:
	# make predictions and visualize the results
	make_predictions(unet, itnet, path)
	



def Compplots(noisyImgB, cleanImgB, best,
	  noisyImgW,cleanImgW,worst, modelName):
	# initialize our figure
	be = ssim(cleanImgB[0][0], best, data_range = best.max()-best.min())
	we = ssim(cleanImgW[0][0], worst, data_range = worst.max()-worst.min())
	bfdkE = ssim(cleanImgB[0][0], noisyImgB[0][0], data_range = noisyImgB[0][0].max()-noisyImgB[0][0].min())
	wfdkE = ssim(cleanImgW[0][0], noisyImgW[0][0], data_range = noisyImgW[0][0].max()-noisyImgW[0][0].min())

	
	figure, ax = plt.subplots(nrows=2, ncols=3, figsize=(12, 10))

	b = ax[0,1].imshow(noisyImgB[0][0].T, vmin=0,vmax=2.5)
	a = ax[0,0].imshow(cleanImgB[0][0].T, vmin=0,vmax=2.5)
	c = ax[0,2].imshow(best.T, vmin=0,vmax=2.5)
	# set the titles of the subplots
	ax[0,0].set_title("Predicted Image (FDK), SSIM = {:.3f}".format(bfdkE))
	ax[0,1].set_title("Clean Image")
	ax[0,2].set_title("Predicted Image (Best), SSIM = {:.3f}".format(be))
	ax[1,0].set_title("Predicted Image (FDK), SSIM = {:.3f}".format(wfdkE))
	ax[1,1].set_title("Clean Image")
	ax[1,2].set_title("Predicted Image (Worst), SSIM = {:.3f}".format(we))
	# set the layout of the figure and display it
	add_zoom_bubble(a)
	add_zoom_bubble(b)
	add_zoom_bubble(c)
	e = ax[1,1].imshow(noisyImgW[0][0].T, vmin=0,vmax=2.5)
	d = ax[1,0].imshow(cleanImgW[0][0].T, vmin=0,vmax=2.5)
	f = ax[1,2].imshow(worst.T, vmin=0,vmax=2.5)
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
	
	plt.savefig("/home/obc22/aitomotools/AItomotools/BestvWorstComparison{}.png".format(modelName))
	

def compPred(unet, itnet, modelName, bestIdx, worstIdx):
	# set model to evaluation mode
	unet.eval()
	itnet.eval()
	# turn off gradient tracking
	vg = ts.volume(shape=(1, *(512,512)), size=(300/512, 300, 300))
	pg = ts.cone(
        angles=360, shape=(1, 900), size=(1, 900), src_orig_dist=575, src_det_dist=1050)
	A = ts.operator(vg, pg)

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
		validateNo = int(np.round(noPatients * 0.1))
		testNo = noPatients - trainNo - validateNo

		trainList = subDirList[:trainNo]
		validateList = subDirList[trainNo:trainNo+validateNo]
		testList = subDirList[trainNo+validateNo:]

		trainDS = loadData(imgPaths=testList, outputSino=True)
		trainLoader = DataLoader(trainDS, shuffle=False,
		batch_size=config.ITNET_BATCH_SIZE, pin_memory=False)
		for (i, (x, y,z)) in enumerate(trainLoader):
			print(i)
			if i == bestIdx:
				bestX = x.to(dev)
				bestY = y.to(dev)
			elif i == worstIdx:
				worstX = x.to(dev)
				worstY = y.to(dev)
			

		cleanImgB = bestY.cpu().numpy()
		cleanImgW = worstY.cpu().numpy()

		noisyImgB = torch.zeros_like(bestY)
		noisyImgW = torch.zeros_like(worstY)
		for j in range(bestX.shape[0]):
			noisyImgB[j]=fdk(A, bestX[j])
			noisyImgW[j]=fdk(A, worstX[j])

		TVpredImgB = tvMin(bestX).forward().cpu().numpy()
		TVpredImgW = tvMin(worstX).forward().cpu().numpy()
		
		UpredImgB = unet(noisyImgB).squeeze().cpu().numpy()
		UpredImgW = unet(noisyImgW).squeeze().cpu().numpy()
		
		IpredImgB = itnet(bestX).squeeze().cpu().numpy()
		IpredImgW = itnet(worstX).squeeze().cpu().numpy()
		
		if modelName == "UNet":
			best = UpredImgB
			worst = UpredImgW
		elif modelName == "ItNet":
			best = IpredImgB
			worst = IpredImgW
		elif modelName == "TV":
			best = TVpredImgB
			worst = TVpredImgW

		noisyImgB = noisyImgB.cpu().numpy()
		noisyImgW = noisyImgW.cpu().numpy()
		Compplots(noisyImgB, cleanImgB, best,
	    noisyImgW, cleanImgW, worst, modelName)
		return



def plotWorst(noisyImgW, cleanImgW, TVpredImgW, UpredImgW, IpredImgW):
	# initialize our figure
	fdkE = ssim(cleanImgW[0][0], noisyImgW[0][0], data_range = noisyImgW[0][0].max()-noisyImgW[0][0].min())
	tvE = ssim(cleanImgW[0][0], TVpredImgW[0], data_range = TVpredImgW[0].max()-TVpredImgW[0].min())
	uE = ssim(cleanImgW[0][0], UpredImgW, data_range = UpredImgW.max()-UpredImgW[0].min())
	iE = ssim(cleanImgW[0][0], IpredImgW, data_range = IpredImgW.max()-IpredImgW[0].min())

	
	figure, ax = plt.subplots(nrows=2, ncols=3, figsize=(12, 10))

	b = ax[0,1].imshow(cleanImgW[0][0].T, vmin=0,vmax=2.5)
	#a = ax[0,0].imshow(noisyImgW[0].T, vmin=0,vmax=2.5)
	c = ax[0,2].imshow(noisyImgW[0][0].T, vmin=0,vmax=2.5)
	# set the titles of the subplots
	#ax[0,0].set_title("Predicted Image (FDK), SSIM=",bfdkE)
	ax[0,1].set_title("Clean Image")
	ax[0,2].set_title("Predicted Image (FDK), SSIM = {:.3f}".format(fdkE))
	ax[1,0].set_title("Predicted Image (TV Min), SSIM = {:.3f}".format(tvE))
	ax[1,1].set_title("Predicted Image (UNet), SSIM = {:.3f}".format(uE))
	ax[1,2].set_title("Predicted Image (ItNet), SSIM = {:.3f}".format(iE))
	# set the layout of the figure and display it
	add_zoom_bubble(b)
	add_zoom_bubble(c)
	d = ax[1,0].imshow(TVpredImgW[0].T, vmin=0,vmax=2.5)
	e = ax[1,1].imshow(UpredImgW.T, vmin=0,vmax=2.5)
	f = ax[1,2].imshow(IpredImgW.T, vmin=0,vmax=2.5)
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
	
	plt.savefig("/home/obc22/aitomotools/AItomotools/WorstComparison.png")
	

def worstComp(unet, itnet, worstIdx):
	# set model to evaluation mode
	unet.eval()
	itnet.eval()
	# turn off gradient tracking
	vg = ts.volume(shape=(1, *(512,512)), size=(300/512, 300, 300))
	pg = ts.cone(
        angles=360, shape=(1, 900), size=(1, 900), src_orig_dist=575, src_det_dist=1050)
	A = ts.operator(vg, pg)

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
		validateNo = int(np.round(noPatients * 0.1))
		testNo = noPatients - trainNo - validateNo

		trainList = subDirList[:trainNo]
		validateList = subDirList[trainNo:trainNo+validateNo]
		testList = subDirList[trainNo+validateNo:]

		trainDS = loadData(imgPaths=testList, outputSino=True)
		trainLoader = DataLoader(trainDS, shuffle=False,
		batch_size=config.ITNET_BATCH_SIZE, pin_memory=False)
		for (i, (x, y)) in enumerate(trainLoader):
			if i == worstIdx:
				worstX = x.to(dev)
				worstY = y.to(dev)
			

		cleanImgW = worstY.cpu().numpy()

		noisyImgW = torch.zeros_like(worstY)
		for j in range(worstX.shape[0]):
			noisyImgW[j]=fdk(A, worstX[j])

		TVpredImgW = tvMin(worstX).forward().cpu().numpy()
		
		UpredImgW = unet(noisyImgW).squeeze().cpu().numpy()
		
		IpredImgW = itnet(worstX).squeeze().cpu().numpy()


		noisyImgW = noisyImgW.cpu().numpy()
		plotWorst(noisyImgW, cleanImgW, TVpredImgW, UpredImgW, IpredImgW)
		return