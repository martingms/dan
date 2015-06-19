import csv
import numpy as np

def load_data(training_path, validation_path):
    training_csv = process_csv(training_path)
    training_data = (training_csv[0][:16937], training_csv[1][:16937])
    validation_data = (training_csv[0][16937:], training_csv[1][16937:])
    test_data = process_csv(validation_path)

    return (training_data, validation_data, test_data)

def process_csv(path):
    data = np.loadtxt(path, skiprows=1, delimiter=',')
    input = data[:,:520]
    coords = data[:,[520,521]]

    # Setting the WAPs that are not present to -105 instead of 100
    # TODO/FIXME: This is just a test, probably stupid.
    input[input > 99.0] = -105
    # Scaling inputs to [0,1]
    input += 105
    input = input / 105.0

    # Normalizing coordinates.
    mean_coords = np.mean(coords, axis=0)
    coord_std =  np.std(coords, axis=0)

    coords = (coords - mean_coords) / coord_std

    return (input, coords)
