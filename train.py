#!/usr/bin/python3
import sys
import json
import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

LEARNING_RATE=.1
ITERATIONS=100000
FEATURE='km'
LABEL='price'


#
# STATS TOOLS
#

def coefficient_of_determination(y, y_pred):
    """
    https://www.ncl.ac.uk/webtemplate/ask-assets/external/maths-resources/statistics/regression-and-correlation/coefficient-of-determination-r-squared.html
    """
    ssr = sum([(y[i] - y_pred[i]) ** 2 for i, _ in enumerate(y)])
    mean_y = mean(y)
    sst = sum([(y[i] - mean_y) ** 2 for i, _ in enumerate(y)])
    return (1 - (ssr/sst))

def mean(lst: [float]):
    """returns the mean value of a list"""
    return (sum(lst) / len(lst))

def mean_squared_error(y: [float], y_pred: [float]):
    """
    cost function, aka L2 loss
    mean squared error is SUM of the squared difference between actual and
    predicted elements
    Source for understanding MSE:
    https://www.britannica.com/science/mean-squared-error
    """
    return (sum([(y[i] - y_pred[i]) ** 2 for i in range(len(y))]) / len(y))


def mean_absolute_error(y: [float], y_pred: [float]):
    """
    another cost function, aka L1 loss
    mean absolute error is SUM of the absolute difference between actual and
    predicted elements
    """
    return (sum([abs(y[i] - y_pred[i]) for i in range(len(y))]) / len(y))


def error(y: [float], y_pred: [float]):
    """
    returns a list with the error difference on each point
    """
    return ([(y_pred[i] - y[i]) for i in range(len(y))])


def normalize_minmax(lst: [float]):
    """normalize data using minmax"""
    x_min = min(lst)
    x_max = max(lst)
    norm = [(x - x_min) / (x_max - x_min) for x in lst ]
    return (norm)


def reverse_slope_intercept(slope: float, intercept: float, original_x: list, original_y: list):
    """Reverses min-max normalization for slope and intercept."""
    x_min = min(original_x)
    x_max = max(original_x)
    y_min = min(original_y)
    y_max = max(original_y)
    slope_new = slope * (y_max - y_min) / (x_max - x_min)
    intercept_new = intercept * (y_max - y_min) + y_min - slope_new * x_min
    return slope_new, intercept_new


def predict_y(slope: float, intercept: float, x: [float]) -> [float]:
    """returns a predicted Y for x values"""
    return ([intercept + slope * xx for xx in x])

#
# GRADIENT DESCENT
#
def gradient_descent(X: [float], y: [float]):
    """
    theta0: intercept
    theta1: slope
    """
    theta = [0, 0]
    info = []
    info.append({"theta0": theta[0], "theta1": theta[1]})
    prev_me = 0
    for i in range(ITERATIONS):
        estimated_y = predict_y(theta[1], theta[0], X)
        err = error(y, estimated_y)
        me = mean_squared_error(y, estimated_y)
        tmptheta0 = LEARNING_RATE * (sum(err) / len(y))
        tmptheta1 = LEARNING_RATE * (sum([e * X[i] for i, e in enumerate(err)]) / len(X))
        theta[0] = theta[0] - tmptheta0
        theta[1] = theta[1] - tmptheta1
        info.append({"theta0": theta[0], "theta1": theta[1]})
        if (prev_me == me):
            break
        prev_me = me
    return(info)


#
# GRAPH FUNCTIONS
#

def set_labels(axs):
    axs[0][0].set_title('Linear Regression')
    axs[0][0].set_xlabel('km')
    axs[0][0].set_ylabel('price')
    axs[1][0].set_title('Linear Regression (normalized)')
    axs[1][0].set_xlabel('km')
    axs[1][0].set_ylabel('price')
    axs[0][1].set_ylabel('Mean Error')
    axs[0][1].set_xlabel('iterations (log)')
    axs[1][1].set_ylabel('Mean Error')
    axs[1][1].set_xlabel('Theta1')


def draw_line(ax, X: [float], y:[float], thetas: (float,float)):
    pred_y = predict_y(thetas[0], thetas[1], X)
    line = ax.plot(X, pred_y, color='lightcoral')
    return(line)


def draw_graphs(fig, axs, info, idx, X, y, X_norm, y_norm):
    slopes = [ info[z]['theta1'] for z in range(idx + 1) ]
    intercepts = [info[z]['theta0'] for z in range(idx + 1)]
    me = [ mean_squared_error(y_norm, predict_y(slopes[i], intercepts[i], X_norm)) for i, _ in enumerate(slopes)]
    coef_det = coefficient_of_determination(y_norm, predict_y(slopes[-1], intercepts[-1], X_norm))
    fig.suptitle(f"Gradient Descent. Iteration: {idx}. Slope: {info[idx]['theta1']}. Intercept: {info[idx]['theta0']}. Mean Error: {me[-1] if me else 0 }. R2: {coef_det}")
    l1 = draw_line(axs[1][0], X_norm, y_norm, (info[idx]['theta1'], info[idx]['theta0']))
    l2 = draw_line(axs[0][0], X, y, reverse_slope_intercept(info[idx]['theta1'], info[idx]['theta0'], X, y))    
    l3 = axs[0][1].plot([z for z in range(len(me))], me, color='mediumorchid')
    l4 = axs[1][1].plot(slopes, me, color='crimson')
    l5 = axs[1][1].plot(intercepts, me, color='teal')
    return [l1, l2, l3, l4, l5]

def graph_it(info, X, y, X_norm, y_norm):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axs = plt.subplots(2,2)
    for i in range(3):
        fig.suptitle(f'Gradient Descent. Starting in {3 - i} seconds.')
        plt.pause(1)
    prt_base, prt_speed = (1, 1)
    axs[0][0].scatter(X, y, s=2, color="steelblue")
    axs[1][0].scatter(X_norm, y_norm, s=2, color="steelblue")
    axs[0][1].set_xscale('log')
    set_labels(axs)
    for idx, i in enumerate(info):
        if not idx % prt_base:
            if not idx % prt_speed:
                prt_speed *= 2
                prt_base +=prt_speed
            lines = draw_graphs(fig, axs, info, idx, X, y, X_norm, y_norm)
            plt.pause(0.001)
            for line in lines:
                line.pop(0).remove()
    draw_graphs(fig, axs, info, len(info) - 1, X, y, X_norm, y_norm)
    plt.show()


#
# INPUT VALIDATION AND DATA LOADING/SAVING
#
def usage():
    """prints usage message"""
    commandline = sys.argv[0]
    if not commandline.startswith("./"):
        commandline = "./" + commandline
    print(commandline, "file_name")


def load_file(filename: str) -> ([float], [float]):
    """loads the data file"""
    X = []
    y = []
    with open(filename) as file:
        reader = csv.DictReader(file, delimiter=',')
        for row in reader:
            X.append(float(row[FEATURE]))
            y.append(float(row[LABEL]))
    return (X, y)


def save_model(info):
    """
    saves the model.
    """
    try:
        with open("model.sav", "w") as file:
            json.dump(info, file)
    except Exception:
        raise ValueError("Could not save into file")
           

def validate_arguments() -> float:
    """no arguments must be passed"""
    if (len(sys.argv) > 2):
        raise AssertionError("Too many arguments")
    elif (len(sys.argv) < 2):
        raise AssertionError("Too few arguments")


def main():
    """validates the input, loads the model and prints the output"""
    try:
        validate_arguments()
        X, y = load_file(sys.argv[1])
        X_norm = normalize_minmax(X)
        y_norm = normalize_minmax(y)
        info = gradient_descent(X_norm, y_norm)
        result = info[-1]
        print(result['theta0'], result['theta1'])
        save_model(result)
        #graph_it(info, X, y, X_norm, y_norm)

    except (AssertionError, ValueError) as error:
        print("Error:", error)
        if type(error) == AssertionError:
            usage()
        exit(1)


if __name__ == '__main__':
    """executes if not loaded as a module"""
    main()
