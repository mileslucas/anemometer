from astropy.io import fits
import numpy as np
import proplot as plot

data = fits.getdata("coadded_and_subtracted_cube.fits")
corrs = fits.getdata("correllelagrams.fits")
indices = range(39, 49)

dataframes = data[indices]
corrsframes = corrs[indices, 24:75, 24:75]

dataflat = np.hstack([fr for fr in dataframes])
corrsflat = np.hstack([fr for fr in corrsframes])

# recreate power stretch
a = 1000
corrspower = (a**corrsflat - 1) / a

fig, axs = plot.subplots(figsize=("510px", "140px"), nrows=2, hspace=0)
ymax = corrspower.shape[0]
ymid = ymax//2
xmax = corrsflat.shape[1]
xmid = corrsframes.shape[-1]//2

axs[0].imshow(dataflat, cmap="inferno", origin="lower", extent=(-0.5, 509.5, -0.5, 50.5))
axs[0].format(xlim=(0, xmax), ylim=(0, ymax))

axs[1].imshow(corrspower, cmap="inferno", origin="lower")
axs[1].hlines(ymid, 0, xmax, color="white", ls="--", lw=0.2)
xs = range(xmid, corrsflat.shape[1], corrsframes.shape[-1])
axs[1].vlines(xs, 0, ymax, color="white", ls="--", lw=0.2)

axs[1].format(xlim=(0, xmax), ylim=(0, ymax))

axs.format(xticks=[], yticks=[], suptitle="N=0; Gaussian(1, 5, 5) filter", fontsize=8)
fig.savefig("mosaic.png")
