import csv


def plot_logs(filename):

    with open(filename) as file:
        plots = csv.reader(file, delimiter = ',') 