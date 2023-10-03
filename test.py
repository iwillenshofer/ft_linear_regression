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
            slope = saved_model["slope"]
            intercept = saved_model["intercept"]
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
        print("The predicted price is:", intercept + slope * mileage)
    except Exception:
        print("Could not convert input to float")



def validate_arguments() -> float:
    """no arguments must be passed"""
    val = 0.0
    if (len(sys.argv) > 1):
        raise AssertionError("Too many arguments")


def signal_handler(sig, frame):
    """signal handler for CTRL+C"""
    print("\nExiting...")
    sys.exit(0)


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
        if type(error) == AssertionError:
            usage()
        exit(1)


if __name__ == '__main__':
    """executes if not loaded as a module"""
    main()
