import csv

# 48 cities 33523
with open("att48-csv.csv", "r") as f:  # creates a list with just the names of the cities
    CITY_NAMES = str((next(csv.reader(f)))[0]).split(";")  # list of all the city names
    DISTANCE = 33523
    PROBLEM ="att48"

with open("att48-csv.csv", "r") as f:
    DATA = list(csv.reader(f, delimiter=';'))  # a table with all distances

# # 26 cities 937
# with open("fri26-csv.csv", "r") as f:  # creates a list with just the names of the cities
#     CITY_NAMES = str((next(csv.reader(f)))[0]).split(";")  # list of all the city names
#     DISTANCE = 937
#     PROBLEM = "fri26"


# with open("fri26-csv.csv", "r") as f:
#     DATA = list(csv.reader(f, delimiter=';'))  # a table with all distances


# # 17 cities; 2085
# with open("gr17-csv.csv", "r") as f:  # creates a list with just the names of the cities
#     CITY_NAMES = str((next(csv.reader(f)))[0]).split(";")  # list of all the city names
#     DISTANCE = 2085
#     PROBLEM = "gr17"

# with open("gr17-csv.csv", "r") as f:
#     DATA = list(csv.reader(f, delimiter=';'))  # a table with all distances

