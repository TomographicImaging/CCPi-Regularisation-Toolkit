import numpy as np

outfile = open("test_imageLena.bin", "wb")  # Open a file for binary write
outfile.write(Im)  # Write it
outfile.flush()  # Optional but a good idea
outfile.close()
# %%
# Define width and height
w, h = 256, 256
# Read file using numpy "fromfile()"
with open("my_file.bin", mode="rb") as f:
    d = np.fromfile(f, dtype=np.float32, count=w * h).reshape(h, w)
