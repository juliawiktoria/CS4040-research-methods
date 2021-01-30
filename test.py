import numpy
import matplotlib.pyplot as plt

PROBLEM = "gr17"

def plot_error(rsm, iem, idm, oem):
    x_ind = numpy.arange(4)
    x_axis = ("RSM", "IEM", "IDM", "OEM")
    err_arr = [rsm, iem, idm, oem]

    plt.xticks(2 * x_ind, x_axis)

    err_bar = plt.bar(2 * x_ind, err_arr, width=0.4, color="g", label="best", zorder=3)

    for bar in err_bar:
        height = bar.get_height()
        plt.annotate('{}'.format(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=8, rotation=0)

    plt.title("GA runtimes over 20 executions of the algorithm for {}".format(PROBLEM))
    plt.legend(loc="upper right")
    plt.ylabel("Time in miliseconds [ms]")
    plt.grid(True, zorder=0)
    plt.show()

plot_error(6.89, 11.56, 9.43, 7.77)