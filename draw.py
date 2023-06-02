import sys
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')


def read_txt(fn):
    fo = open(fn, 'r')

    val_acc = []
    dp = []
    eo = []

    for line in fo.readlines():
        line = line.strip()

        if 'val_accuracy' in line:
            val_acc.append(float(line.split(':')[-1]))
        elif 'demographic parity' in line:
            dp.append(float(line.split(':')[-1]))
        elif 'equal opportunity' in line:
            eo.append(float(line.split(':')[-1]))

    fo.close()

    return val_acc, dp, eo


if __name__ == "__main__":
    val_acc_bl = []
    dp_bl = []
    eo_bl = []

    val_acc = []
    dp = []
    eo = []

    val_acc_bl, dp_bl, eo_bl = read_txt(sys.argv[1])
    val_acc_bl = val_acc_bl[4::4]
    dp_bl = dp_bl[4::4]
    eo_bl = eo_bl[4::4]

    for i in range(2, len(sys.argv)):
        v, d, e = read_txt(sys.argv[i])
        val_acc += v
        dp += d
        eo += e

    plt.scatter(dp, val_acc, marker='.')
    plt.scatter(dp_bl, val_acc_bl, marker='.')
    # plt.ylim(0.7, 0.725)
    plt.savefig('logs/vanilla.jpg')
