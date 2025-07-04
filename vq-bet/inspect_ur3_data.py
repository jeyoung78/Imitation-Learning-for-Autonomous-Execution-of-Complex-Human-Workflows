import numpy as np

# adjust this path as needed:
DATA_DIR = r"C:\Users\jupar\Downloads\vqbet_datasets_for_release\vqbet_datasets_for_release\ur3"

for name in ("data_act.npy", "data_msk.npy", "data_obs.npy"):
    path = f"{DATA_DIR}\\{name}"
    arr  = np.load(path)
    print(f"\n=== {name} ===")
    print(" shape:",   arr.shape)
    print(" dtype:",   arr.dtype)
    print(" min/max:", arr.min(), "/", arr.max())
    print(" sample:",  arr.flatten()[:20], "…")  # first 20 entries
    # if you really want to see everything, uncomment the next line:
    # print(arr)
