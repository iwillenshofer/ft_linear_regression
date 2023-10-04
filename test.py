#!/usr/bin/python3
import sys
import json
import signal


# JSON LOADING
def load_model() -> (float, float):
    """
    loads the model.
    raises exception if not valid.
    returns (0, 0) if doesn't exist
    returns slope, intercept if succeeded
    """
    try:
        with open("model.sav", "r") as file:
            saved_model = json.load(file)
            slope = saved_model["theta1"]
            intercept = saved_model["theta0"]
            return (float(slope), float(intercept))
    except FileNotFoundError:
        return (0, 0)
    except Exception:
        raise ValueError("Invalid model file")


# INPUT VALIDATION
def usage():
    """prints usage message"""
    commandline = sys.argv[0]
    if not commandline.startswith("./"):
        commandline = "./" + commandline
    print("Input must be a car_mileage (Positive Float or Integer)")


def input_loop(slope: float, intercept: float):
    """validates user input, returning its value as float if valid"""
    try:
        mileage = float(input("Type in the car mileage: "))
        if mileage < 0:
            raise AssertionError("Input must be positive")
        result = intercept + slope * mileage
        print("The predicted price is:", '$%.2f' % result if result > 0 else 0.00)
    except AssertionError as error:
        print(error)
    except Exception:
        print("Could not convert input to float")
    input_loop(slope, intercept)


def validate_arguments() -> float:
    """no arguments must be passed"""
    if (len(sys.argv) > 1):
        raise AssertionError("Too many arguments")


def signal_handler(sig, frame):
    """signal handler for CTRL+C"""
    print("\nExiting...")
    exit(0)


# MAIN
def main():
    """validates the input, loads the model and prints the output"""
    signal.signal(signal.SIGINT, signal_handler)
    try:
        validate_arguments()
        slope, intercept = load_model()
        input_loop(slope, intercept)
    except (AssertionError, ValueError) as error:
        print("Error:", error)
        if type(error) is AssertionError:
            usage()
        exit(1)


if __name__ == '__main__':
    """executes if not loaded as a module"""
    main()
