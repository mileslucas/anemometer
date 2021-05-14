import argparse
import logging
import os
import sys
import time
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from skimage.registration import phase_cross_correlation
from pyMilk.interfacing.isio_shmlib import SHM


plt.style.use("ggplot")

# set up logger
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

# define command line args
parser = argparse.ArgumentParser()
parser.add_argument(
    "-N",
    help="Measure every `N` frames. Minimum measurable wind-speed is (200 m/s) / N",
    default=200,
    type=int,
)
# parser.add_argument(
#     "-i", "--inject", help="Inject fake signal", action="store_true", default=False
# )

# constants
RATE = 2e3  # 2 kHz
PYDM_PLATESCALE_X = 45 / 8.2 # px/m in DM space
PYDM_PLATESCALE_Y = 44.8 / 8.2 # px/m in DM space
PYDM_SPEEDSCALE_X = RATE * PYDM_PLATESCALE_X
PYDM_SPEEDSCALE_Y = RATE * PYDM_PLATESCALE_Y




fig_frames, frames = plt.subplots(ncols=2, figsize=(6, 4))
frames[0].set_title("N")
frames[1].set_title("N + 1")
[fr.grid(False) for fr in frames]
frame1 = frames[0].imshow(np.zeros((60, 60)), origin="lower", cmap="inferno")
frame2 = frames[1].imshow(np.zeros((60, 60)), origin="lower", cmap="inferno")
plt.tight_layout()


def measure_shift(previous_frame, current_frame, mask=None, f=100):
    # note frames are reversed so the shift measures wind speed
    # in terms of the reference image
    shift_yx = phase_cross_correlation(
        current_frame,
        previous_frame,
        return_error=False,
        reference_mask=mask,
        upsample_factor=f  # will have no effect if mask is not None
    )
    x = shift_yx[1] * PYDM_SPEEDSCALE_X
    y = shift_yx[0] * PYDM_SPEEDSCALE_Y
    r, theta = polar_transform(x=x, y=y)
    return x, y, r, theta

# def inject_line(frame, i=0, dx=4 , dy=6, n_coadd=100, amp=1):
#     """
#     create a streak with high S/N. This will look like a hot-pixel has been smeared over `n_coadds`
#     """
#     dx *= PYWFS_PLATESCALE / PYWFS_RATE
#     dy *= PYWFS_PLATESCALE / PYWFS_RATE
#     # get starting point and ending point for bottom-left aperture
#     x0 = (i * dx) % 60
#     y0 = (i * dy) % 60
#     x1 = x0 + n_coadd * dx
#     y1 = y0 + n_coadd * dy

#     for x, y in zip(np.linspace(x0, x1, n_coadd), np.linspace(y0, y1, n_coadd)):
#         i = int(x)
#         j = int(y)
#         frame[i, j] = amp  # bottom-left
#         frame[i, min(119, j + 60)] = amp  # top-left
#         frame[min(119, i + 60), j] = amp # bottom-right
#         frame[min(119, i + 60), min(119, j + 60)] = amp # bottom-left

#     return frame

def polar_transform(x, y):
    r = np.hypot(x, y)
    theta = np.rad2deg(np.arctan2(y, x))
    if theta < 0:
        theta += 360
    return r, theta

def get_mask():
    # use maps calculated from aol0_dmmap
    dmmask = SHM("aol0_dmmask").get_data(check=False)
    dmslaved = SHM("aol0_dmslaved").get_data(check=False)
    mask = dmmask.astype(bool) ^ dmslaved.astype(bool) # XOR
    return mask

def loop_offset(N, plot=True, maxiter=np.inf, **kwargs):
    # stream = SHM("aol0_modeval_ol")
    stream = SHM("dm00disp10")
    basis = SHM("aol0_DMmodes").get_data(check=False)

    mask = get_mask()

    offsets = []

    if plot:
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].set_xlabel(r"$\Delta x$")
        axs[0, 1].set_xlabel(r"$\Delta y$")
        axs[1, 0].set_xlabel(r"$\Delta r$")
        axs[1, 1].set_xlabel(r"$\theta$")
        plt.tight_layout()

    i = 0
    # previous_modes = stream.get_data(check=True) # make sure to get data so semID exists
    # previous_frame = basis @ previous_modes
    logger.info("starting loop")
    while i < maxiter:
        # previous_frame = stream.get_data(check=True)
        # # wait for N frames (TODO async)
        # stream.IMAGE.semflush(stream.semID)
        # stream.IMAGE.semwait(stream.semID)
        # cnt = stream.IMAGE.md.cnt0
        # print(cnt)
        # for _ in range(N - 1 - (cnt % N)):
        #     stream.IMAGE.semwait(stream.semID)

        # current_modes = stream.get_data(check=True)

        # # calculate open-loop reconstruction
        # current_frame = basis @ current_modes

        # current_frame = stream.get_data(check=True)



        cube = stream.multi_recv_data(N+1, outputFormat=1)
        previous_frame = cube[0]
        current_frame = cube[-1]

        # show frames
        plt.figure(fig_frames.number)
        frames[0].imshow(previous_frame * mask, origin="lower")
        frames[1].imshow(current_frame * mask, origin="lower")
        plt.pause(1e-5)

        # make offset measurement
        x, y, r, theta = measure_shift(previous_frame, current_frame, mask=mask)
        x /= N; y /= N; r /= N # convert to average shift per frame
        logger.info(f"{i:03d}: x={x} m/s, y={y} m/s, r={r} m/s, theta={theta} deg")
        offsets.append(dict(x=x, y=y, r=r, theta=theta))



        # plotting
        if plot:
            axs[0, 0].scatter(i, x, c="C0")
            axs[0, 1].scatter(i, y, c="C1")
            axs[1, 0].scatter(i, r, c="C2")
            axs[1, 1].scatter(i, theta, c="C3")
            # fig.canvas.copy_from_bbox(ax.bbox) # blitting
            plt.figure(fig.number)
            plt.pause(1e-5)

        # loop updates
        i += 1
        previous_frame = current_frame

    return offsets

if __name__ == "__main__":
    # parse args
    args = parser.parse_args()
    logger.debug(f"parsed args: {args}")

    loop_offset(N=args.N)
