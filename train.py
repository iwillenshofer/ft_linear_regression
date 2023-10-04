#!/usr/bin/python3
import sys
import json
import csv
import warnings
import matplotlib.pyplot as plt

LEARNING_RATE = .1
ITERATIONS = 100000
FEATURE = 'km'
LABEL = 'price'
INPUT_FILE = 'data.csv'
OUTPUT_FILE = 'model.sav'


#
# STATS TOOLS
#
def coefficient_of_determination(y, y_pred):
    """calculates the coefficient of determination"""
    ssr = sum([(y[i] - y_pred[i]) ** 2 for i, _ in enumerate(y)])
    mean_y = mean(y)
    sst = sum([(y[i] - mean_y) ** 2 for i, _ in enumerate(y)])
    return (1 - (ssr/sst))


def residual_stdev(y, y_pred):
    """calculates the residual standard deviation"""
    residuals = [(y[i] - y_pred[i]) for i, _ in enumerate(y)]
    mean_residual = sum(residuals) / len(y)
    return (sum([(i - mean_residual) ** 2 for i in residuals])
            / (len(y) - 1)) ** .5


def mean(lst: [float]):
    """returns the mean value of a list"""
    return (sum(lst) / len(lst))


def mean_squared_error(y: [float], y_pred: [float]):
    """
    cost function, aka L2 loss
    mean squared error is SUM of the squared difference between actual and
    predicted elements
    """
    return (sum([(y[i] - y_pred[i]) ** 2 for i in range(len(y))]) / len(y))


def error(y: [float], y_pred: [float]):
    """
    returns a list with the error difference on each point
    """
    return ([(y_pred[i] - y[i]) for i in range(len(y))])


def normalize_minmax(lst: [float]):
    """normalize data using minmax"""
    x_min = min(lst)
    x_max = max(lst)
    norm = [(x - x_min) / (x_max - x_min) for x in lst]
    return (norm)


def reverse_slope_intercept(slope: float, intercept: float,
                            original_x: list, original_y: list):
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
        tmptheta1 = LEARNING_RATE * (
            sum([e * X[i] for i, e in enumerate(err)]) / len(X))
        theta[0] = theta[0] - tmptheta0
        theta[1] = theta[1] - tmptheta1
        info.append({"theta0": theta[0], "theta1": theta[1]})
        if (prev_me == me):
            break
        prev_me = me
    return info


#
# GRAPH FUNCTIONS
#
def set_labels(axs):
    """sets labels for the graphs"""
    axs[0][0].set_title('Linear Regression')
    axs[0][0].set_xlabel('km')
    axs[0][0].set_ylabel('price')
    axs[1][0].set_title('Linear Regression (normalized)')
    axs[1][0].set_xlabel('km')
    axs[1][0].set_ylabel('price')
    axs[0][1].set_ylabel('MSE')
    axs[0][1].set_xlabel('iterations (log)')
    axs[1][1].set_ylabel('MSE')
    axs[1][1].set_xlabel('Theta values')


def draw_line(ax, X: [float], thetas: (float, float)):
    """draws the regression line"""
    pred_y = predict_y(thetas[0], thetas[1], X)
    line = ax.plot(X, pred_y, color='lightcoral')
    return line


def draw_graphs(fig, axs, info, idx, X, y, X_norm, y_norm):
    """draws graph"""
    slopes = [info[z]['theta1'] for z in range(idx + 1)]
    intercepts = [info[z]['theta0'] for z in range(idx + 1)]
    me = [mean_squared_error(
        y_norm, predict_y(slopes[i], intercepts[i], X_norm)
        ) for i, _ in enumerate(slopes)]
    coef_det = coefficient_of_determination(
        y_norm, predict_y(slopes[-1], intercepts[-1], X_norm))
    fig.suptitle(f"Gradient Descent. Iteration: {idx}. "
                 + "$\\theta1_{norm}: " + ('%.4f' % info[idx]['theta1'])
                 + ". \\theta0_{norm}: " + ('%.4f' % info[idx]['theta0'])
                 + ". \\theta1_{real}: " + ('%.4f' % info[idx]['otheta1'])
                 + ". \\theta0_{real}$: " + ('%.4f' % info[idx]['otheta0'])
                 + f". MSE: {'%.4f' % me[-1] if me else 0 }."
                 + f". R2: {'%.4f' % coef_det}.")
    l1 = draw_line(axs[1][0], X_norm,
                   (info[idx]['theta1'], info[idx]['theta0']))
    l2 = draw_line(axs[0][0], X,
                   reverse_slope_intercept(
                       info[idx]['theta1'], info[idx]['theta0'], X, y))
    l3 = axs[0][1].plot([z for z in range(len(me))], me, color='mediumorchid')
    l4 = axs[1][1].plot(slopes, me, color='crimson', label='theta1')
    l5 = axs[1][1].plot(intercepts, me, color='teal', label='theta0')
    axs[1, 1].legend(loc="upper left")
    return [l1, l2, l3, l4, l5]


def graph_it(info, X, y, X_norm, y_norm):
    """loops through all iteractions to draw graphs"""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axs = plt.subplots(2, 2)
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
                prt_base += prt_speed
            lines = draw_graphs(fig, axs, info, idx, X, y, X_norm, y_norm)
            plt.pause(0.001)
            for line in lines:
                line.pop(0).remove()
    draw_graphs(fig, axs, info, len(info) - 1, X, y, X_norm, y_norm)
    plt.show()


#
# INFORMATION ON LINEAR REGRESSION
#
def display_infos(info, X, y, X_norm, y_norm):
    pred_y = predict_y(info[-1]['otheta1'], info[-1]['otheta0'], X)
    print(f"Max iterations: {ITERATIONS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"𝜃₁: {info[-1]['otheta1']}")
    print(f"𝜃₀: {info[-1]['otheta0']}")
    print(f"iterations: {len(info)}")
    print(f"Residual StdDev: {residual_stdev(y, pred_y)}")
    print(f"R²: {coefficient_of_determination(y, pred_y)}")


#
# FILE HANDLING
#
def load_file(filename: str) -> ([float], [float]):
    """loads the data file"""
    X = []
    y = []
    try:
        with open(filename) as file:
            reader = csv.DictReader(file, delimiter=',')
            for row in reader:
                X.append(float(row[FEATURE]))
                y.append(float(row[LABEL]))
    except KeyError:
        raise(ValueError(f"File does not contain {FEATURE} or {LABEL} rows"))
    except Exception:
        raise(ValueError(f"Could not read data file '{filename}'"))
    return (X, y)


def save_model(info):
    """
    saves the model.
    """
    try:
        with open(OUTPUT_FILE, "w") as file:
            json.dump(info, file)
    except Exception:
        raise ValueError("Could not save into file")


#
# INPUT VALIDATION
#
def usage():
    """prints usage message"""
    commandline = sys.argv[0]
    if not commandline.startswith("./"):
        commandline = "./" + commandline
    print(commandline, "[--graph|--help]")


def validate_arguments():
    """
    validate args returns a list of options
    """
    valid_options = ['--graph', '--info']
    for i in sys.argv[1:]:
        if i == '--help':
            usage()
            exit(0)
        elif i not in valid_options:
            raise AssertionError(f"unknown option: {i}")
    return [i for i in sys.argv[1:] if i in valid_options]


def main():
    """validates the input, loads the model and prints the output"""
    warnings.filterwarnings("ignore")
    try:
        options = validate_arguments()
        X, y = load_file(INPUT_FILE)
        X_norm = normalize_minmax(X)
        y_norm = normalize_minmax(y)
        info = gradient_descent(X_norm, y_norm)
        for i in info:
            i["otheta1"], i["otheta0"] = reverse_slope_intercept(
                i['theta1'], i['theta0'], X, y)
        result = info[-1]
        save_model({"theta1": result['otheta1'], "theta0": result['otheta0']})
        if '--graph' in options:
            graph_it(info, X, y, X_norm, y_norm)
        if '--info' in options:
            display_infos(info, X, y, X_norm, y_norm)
    except (AssertionError, ValueError) as error:
        print("Error:", error)
        if type(error) is AssertionError:
            usage()
        exit(1)


if __name__ == '__main__':
    """executes if not loaded as a module"""
    main()
