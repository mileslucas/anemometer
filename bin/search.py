import os
from pathlib import Path

if __name__ == "__main__":
    root = Path("/mnt/sdata01")
    files = []
    for subdir in root.iterdir():
        try:
            if not subdir.is_dir(): continue
            if not "20" in str(subdir): continue

            for subsubdir in subdir.iterdir():
                if not "aol0_modeval_ol" in str(subsubdir): continue

                filelist = sorted(subsubdir.glob("aol0_modeval_ol*.fits*"), key=os.path.getctime)
                # ~240 cubes per hour at 2kHz
                files.extend([str(f) for f in filelist[::120]])

        except PermissionError:
            continue

    with open("files_to_scan.txt", "w") as fh:
        fh.write("\n".join(files))

