import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    file_prefixes = [
        "benchmarks/baseline",
        "benchmarks/jit",
        "benchmarks/jitpara",
    ]
    suf_mean = "_mean.npy"
    # suf_stad = "_stad.npy"
    sizes = np.logspace(3, 7, 17, base=4).astype(np.int64)

    for ii in range(len(file_prefixes)):
        name = file_prefixes[ii].split("/")[1]
        means = np.load(file_prefixes[ii] + suf_mean)
        # stads = np.load(file_prefixes[ii] + suf_stad)

        mode_names = ["Remake", "Add", "Add use_jac"]
        for mi in [0, 1, 2]:
            if ii == 0 and mi == 0:
                plt.plot(sizes, means[mi], "o", label=mode_names[mi])
                continue
            elif mi == 0:
                continue
            plt.plot(sizes, means[mi], "o", label=name + mode_names[mi])

    plt.xscale("log")
    plt.xlabel("Number of data")
    plt.yscale("log")
    plt.ylabel("Calculating time [s]")
    plt.legend()
    plt.show()