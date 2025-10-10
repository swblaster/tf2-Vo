import numpy as np
from os import listdir
from os.path import isfile, join

input_path = "./t100000"
if __name__ == "__main__":
    files = [f for f in listdir(input_path) if isfile(join(input_path, f))]
    data = []
    labels = []
    for i in range(len(files)):
        path = input_path + "/" + files[i]
        f = open(path, "r")
        lines = f.readlines()
        f.close()
        sample = []
        for j in range (len(lines)):
            line = lines[j].split('\n')
            value = float(line[0])
            sample.append(value)
        data.append(sample)

        tokens = files[i].split('_')
        sigma = tokens[1].split('.')[0]
        label = np.zeros((1000))
        for j in range (1000):
            label[j] = float((int(sigma) - 50) // 50.0)
        labels.append(label)
    samples = np.array(data)
    samples = np.reshape(samples, (14, 1000, 200))

    for i in range (14):
        offset = 1000 * 200 * i
        name = "sigma_" + str(int(labels[i][0])) + ".txt"
        f = open(name, "w")
        for j in range (1000):
            for k in range (200):
                f.write("%16.14f " %(samples[i][j][k]))
            f.write("\n")
        f.close()
