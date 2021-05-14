from astropy.io import fits
import numpy as np
import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("files", nargs="+")
parser.add_argument("-o", "--output", help="name of output fits cube", default="thinned.fits")
parser.add_argument("-n", help="how many frames to thin by", type=int)

if __name__ == "__main__":
    args = parser.parse_args()
    queue = [] 
    for filename in tqdm.tqdm(args.files):
        data = fits.getdata(filename)
        queue.append(data[::args.n])

    cube = np.concatenate(queue, axis=0)
    print(f"output cube size: {cube.shape}")
    fits.writeto(args.output, cube, overwrite=True)

