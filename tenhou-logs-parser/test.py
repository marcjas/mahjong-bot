import numpy as np
from tqdm import tqdm

filename = 'data.bin'
num_samples = 3600000
rows, cols = 30, 32
dtype = np.half

# format: <num_samples> <rows> <cols> <sample0> <sample1>...
with open(filename, 'wb') as fout:
    # write a header that contains the total number of samples and the rows and columns per sample
    fout.write(np.array((num_samples, rows, cols), dtype=np.int32).tobytes())
    for i in tqdm(range(num_samples)):
        # random placeholder
        sample = np.random.randn(rows, cols).astype(dtype)
        # print(len(sample))
        # break
        # write data to file
        fout.write(sample.tobytes())

