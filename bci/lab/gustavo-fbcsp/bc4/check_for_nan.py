# Loop through all data checking if there are NAN values - normally there are none

from struct import unpack
from numpy import isnan, arange
from load_data import load_data


if __name__ == "__main__":

    subjectsT = arange(1, 10)
    classesT = [[1,2], [3,4]]

    for SUBJECT in subjectsT:
        for classes in classesT:
            X = load_data(SUBJECT, classes)
            print any(isnan(X))
