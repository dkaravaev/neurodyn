import numpy as np

class Numbers:
    zero = np.asarray([
        0, 1, 1, 1, 0,
        1, 0, 0, 0, 1,
        1, 0, 0, 0, 1,
        1, 0, 0, 0, 1,
        1, 0, 0, 0, 1,
        0, 1, 1, 1, 0
     ])

    one = np.asarray([
        0, 1, 1, 1, 0,
        0, 0, 0, 1, 0,
        0, 0, 0, 1, 0,
        0, 0, 0, 1, 0,
        0, 0, 0, 1, 0,
        0, 0, 0, 1, 0
     ])

    two = np.asarray([
        1, 1, 1, 1, 0,
        0, 0, 0, 1, 0,
        0, 0, 0, 1, 0,
        0, 1, 1, 1, 0,
        1, 0, 0, 0, 0,
        1, 1, 1, 1, 1
     ])

    three = np.asarray([
        1, 1, 1, 1, 1,
        0, 0, 0, 0, 1,
        1, 1, 1, 1, 1,
        0, 0, 0, 0, 1,
        0, 0, 0, 0, 1,
        1, 1, 1, 1, 1
    ])

    four = np.asarray([
        1, 0, 0, 0, 1,
        1, 0, 0, 0, 1,
        1, 1, 1, 1, 1,
        0, 0, 0, 0, 1,
        0, 0, 0, 0, 1,
        0, 0, 0, 0, 1
    ])

    five = np.asarray([
        1, 1, 1, 1, 1,
        1, 0, 0, 0, 0,
        1, 1, 1, 1, 1,
        0, 0, 0, 0, 1,
        0, 0, 0, 0, 1,
        1, 1, 1, 1, 1
    ])

    patterns = [zero, one, two, three, four, five]

    @staticmethod
    def print(number):
        for row in (number.reshape((6, 5))).tolist():
            print('| ' + ' '.join(' *'[int(val)] for val in row))