from data_file import DATA
from data_file import CITY_NAMES


def get_distance(a, b):
    # finds a distance between two cities

    distance = float(DATA[CITY_NAMES.index(b)+1][CITY_NAMES.index(a)])
    return distance


def calculate_cost(cities):  # "path" here is a tuple
    # calculates the total "cost" of the chosen path

    cost = 0  # default cost before visiting any cities
    # cities = list(cities)  # converting a tuple into a list
    length = len(cities)

    for i in range(0, length - 1):

        cost += get_distance(cities[i], cities[i + 1])

    # adding the distance from the last to the first city
    cost += get_distance(cities[0], cities[length - 1])
    cost = round(cost, 2)
    return cost
