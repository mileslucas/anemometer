import argparse
import logging
import os
import pathlib

from astropy.io import fits
import numpy as np
import tqdm

# logging
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# directories
def configdir(*tokens):
    return os.path.join("/", "home", "scexao", "AOloop", "AOloop0", *tokens)

def get_modes():
    base = pathlib.Path(configdir("DMmodes"))
    bases = {
        1217: fits.getdata(list(base.glob("DMmodes_2020-11-24*.fits*"))[0]),
        1193: fits.getdata(list(base.glob("DMmodes_2021-01-19*.fits*"))[0]),
        1190: fits.getdata(list(base.glob("DMmodes_2021-03-19*.fits*"))[0]),
    }
    return bases

def default_LWE():
    return os.path.join(configdir("mkmodestmp"), "fmodes1_05.fits")

# arg parsing
parser = argparse.ArgumentParser(
    description="generate pseudo-open-loop cubes via mode reconstruction"
)
parser.add_argument("filename", nargs="+")
# parser.add_argument("-m", "--modes", default=default_mask(), help="DM modes file")
parser.add_argument("-o", "--output", default="", help="output directory")
parser.add_argument("-f", "--filter", action="store_true", help="filter out the LWE modes")

if __name__ == "__main__":
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    bases = get_modes()
    if args.filter:
        lwe_frames = fits.getdata(default_LWE()).reshape(-1, 50**2).T

    for filename in tqdm.tqdm(args.filename):
        modevals = np.squeeze(fits.getdata(filename))
        logger.debug(f"loading {modevals.shape} mode values from {filename}")
        mode_basis = bases[modevals.shape[-1]]
        full_frame = np.tensordot(modevals, mode_basis, axes=1)
        if args.filter:
            flat_full_frames = full_frame.reshape(-1, 50**2).T
            lwe_coeffs, _, _, _ = np.linalg.lstsq(lwe_frames, flat_full_frames, rcond=None)
            lwe_recon = np.reshape(lwe_coeffs.T @ lwe_frames.T, full_frame.shape)
            full_frame -= lwe_recon

        outname = os.path.basename(filename).replace("aol0_modeval_ol", "pol_recon")
        # old files saved in compressed, way too slow to save compressed again
        if ".fits.gz" in outname:
            outname = outname.replace(".gz", "")
        outpath = os.path.join(args.output, outname)
        fits.writeto(outpath, full_frame, overwrite=True)
        logger.debug(f"saved file to {outpath}")
