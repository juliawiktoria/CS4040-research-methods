import time
from data_file import CITY_NAMES, DISTANCE, PROBLEM
import numpy
import functions
import random
import statistics
import matplotlib.pyplot as plt
from itertools import chain, zip_longest, groupby
from operator import itemgetter
import pandas as pd
import os.path

POP_SIZE_100 = 20
ITERATIONS = 50
GENERATIONS = 500

PROBABILITY = 0.5

BEST_PATHS_RSM = []
BEST_PATHS_IEM = []
BEST_PATHS_IDM = []
BEST_PATHS_OEM = []
TIMES_RSM =[]
TIMES_IEM =[]
TIMES_IDM =[]
TIMES_OEM =[]

def calc_fitness(population):  # DONE
    fitness_list = []

    for member in population:
        fitness = round(functions.calculate_cost(member), 2)
        fitness_list.append(fitness)

    return fitness_list

def crossover(parents):  # ordered crossover (based on week 2 exercises)

    # 'parents' is a list of two 2-element tuples; parent1 and parent2 are just a simplification
    parent1 = parents[0][0]  # lists of city names
    parent2 = parents[1][0]

    # creating two default children
    child_1 = [None] * len(parent1)
    child_2 = [None] * len(parent1)
    start, end = sorted([random.randrange(len(parent1)) for _ in range(2)])

    child_1[start:end] = parent1[start:end]
    child_2[start:end] = parent2[start:end]

    length = len(parent1)
    p2_ind = end
    c1_ind = end
    while None in child_1:
        if parent2[p2_ind % length] not in child_1:
            child_1[c1_ind % length] = parent2[p2_ind % length]
            c1_ind += 1
        p2_ind += 1

    p1_ind = end
    c2_ind = end
    while None in child_2:
        if parent1[p1_ind % length] not in child_2:
            child_2[c2_ind % length] = parent1[p1_ind % length]
            c2_ind += 1
        p1_ind += 1

    # generates two children and chooses the better one
    if functions.calculate_cost(child_1) < functions.calculate_cost(child_2):
        child = child_1
    else:
        child = child_2

    return child

def mutationFunction(individual, name="reverse"):
    if_mutate = random.uniform(0, 1)

    if if_mutate <= PROBABILITY:
        if name == "reverse":
            individual_length = len(individual)

            mut_len = random.randint(1, individual_length // 2)
            # print("mut len: {}".format(mut_len))

            start_ind = random.randint(0, mut_len - 1)
            end_ind = start_ind + mut_len - 1
            # print("start ind {} and end ind {}".format(start_ind, end_ind))
            # print(individual)
            individual[start_ind:end_ind + 1] = individual[start_ind:end_ind + 1][::-1]
        if name == "exchange":
            individual_length = len(individual)

            mut_len = random.randint(1, individual_length // 2)
            # print("mut len: {}".format(mut_len))

            start_ind = random.randint(0, mut_len - 1)
            end_ind = start_ind + mut_len - 1
            # print("start ind {} and end ind {}".format(start_ind, end_ind))
            # print(individual)
            individual[start_ind:end_ind + 1] = individual[start_ind:end_ind + 1][::-1]

            subtour_ind = []
            for i in range(start_ind, end_ind + 1):
                subtour_ind.append(i)

            idices = []
            for i in range(start_ind):
                idices.append(i)

            for i in range(end_ind + 1, individual_length - 1):
                idices.append(i)
            # print("subtour ind: {}".format(subtour_ind))
            # print("outside ind: {}".format(idices))
            # choose a city in the substring and out the substring and swap them
            swapping_ind_subtour = random.choice(subtour_ind)
            swapping_ind_outside = random.choice(idices)
            individual[swapping_ind_outside], individual[swapping_ind_subtour] = individual[swapping_ind_subtour], individual[swapping_ind_outside]
        if name == "displacement":
            individual_length = len(individual)

            mut_len = random.randint(1, individual_length // 2)
            # print("mut len: {}".format(mut_len))

            start_ind = random.randint(0, mut_len - 1)
            end_ind = start_ind + mut_len - 1
            # print("start ind {} and end ind {}".format(start_ind, end_ind))
            # print(individual)
            individual[start_ind:end_ind + 1] = individual[start_ind:end_ind + 1][::-1]

            subtour_ind = []
            for i in range(start_ind, end_ind + 1):
                subtour_ind.append(i)

            idices = []
            for i in range(start_ind):
                idices.append(i)

            for i in range(end_ind + 1, individual_length - 1):
                idices.append(i)

            # print("subtour ind: {}".format(subtour_ind))
            # print("outside ind: {}".format(idices))

            rand_ind = random.choice(idices)

            if rand_ind < start_ind:
                left_one = individual[0:rand_ind]
                left_two = individual[rand_ind:start_ind]
                subtour = individual[start_ind:end_ind + 1]
                right = individual[end_ind + 1:]
                # print("left one: {}\n left two: {}\n subtour: {}\nright: {}".format(left_one, left_two, subtour, right))
                individual = left_one + subtour + left_two + right

            if rand_ind > end_ind:
                right_one = individual[end_ind + 1:rand_ind]
                right_two = individual[rand_ind:]
                subtour = individual[start_ind:end_ind + 1]
                left = individual[0:start_ind]
                # print("left: {}\n subtour: {}\nright one: {}\n right two: {}".format(left, subtour, right_one, right_two))
                individual = left + right_one + subtour + right_two
        if name == "oddeven":
            individual_length = len(individual)

            mut_len = random.randint(3, individual_length // 2)

            start_ind = random.randint(0, mut_len - 1)
            end_ind = start_ind + mut_len - 1

            subtour = individual[start_ind:end_ind + 1]
            first = []
            second = []
            for i in range(0, len(subtour)):
                if i % 2 == 0:
                    first.append(subtour[i])
                else:
                    second.append(subtour[i])
            first = [first[-1]] + first[:-1]
            second = [second[-1]] + second[:-1]

            new = list(filter(None.__ne__ ,chain.from_iterable(zip_longest(first,second)))) 
            individual[start_ind:end_ind + 1] = new
    return individual

def init_population():  # DONE
    # generates a population of the given size
    starting_path = get_cities()

    population_paths = [list(numpy.random.permutation(starting_path)) for _ in range(POP_SIZE_100)]
    population_fitness = calc_fitness(population_paths)

    # building a list of tuples (path, fitness) sorted in ascending order based on the distance
    population = sorted(list(zip(population_paths, population_fitness)), key=lambda x: x[1])

    return population

def sort_population(population):  # DONE
    population = sorted(population, key=lambda x: x[1])
    return population

def parent_selection(population):  # DONE
    # select two fittest individuals to be parents
    parents = population[:2]
    return parents

def generate_offspring(population, parents, size, mutation):  # DONE
    how_many = size - len(population)

    # generating new offspring (number of children is equal to the size of the population)
    for i in range(0, how_many - 1):
        child = crossover(parents)
        # mutating a child after generating it
        child = mutationFunction(child, mutation)
        # appending newly created and mutated child to the population
        population.append((child, functions.calculate_cost(child)))

    # sorting the population based on the distance
    population = sort_population(population)
    # cutting off the weakest individuals so the population size is the same as in the beginning
    population = population[:size]
    return population

def get_cities():  # DONE
    cities = CITY_NAMES
    return cities

def kill_useless_individuals(pop_list):
    pop_list = pop_list[:len(pop_list)//2]
    return pop_list

def init_algorithm(pop, mutation):
    # population = init_population()
    number_of_generations = 0

    if mutation == "reverse":
        BEST_PATHS_RSM.append((number_of_generations, pop[0][1]))
    elif mutation == "exchange":
        BEST_PATHS_IEM.append((number_of_generations, pop[0][1]))
    elif mutation == "displacement":
        BEST_PATHS_IDM.append((number_of_generations, pop[0][1]))
    elif mutation == "oddeven":
        BEST_PATHS_OEM.append((number_of_generations, pop[0][1]))

    counter = 0
    while counter < GENERATIONS:  # number of generations

        parents = parent_selection(pop)
        pop = kill_useless_individuals(pop)
        pop = generate_offspring(pop, parents, POP_SIZE_100, mutation)
        number_of_generations += 1
        if mutation == "reverse":
            BEST_PATHS_RSM.append((number_of_generations, pop[0][1]))
        elif mutation == "exchange":
            BEST_PATHS_IEM.append((number_of_generations, pop[0][1]))
        elif mutation == "displacement":
            BEST_PATHS_IDM.append((number_of_generations, pop[0][1]))
        elif mutation == "oddeven":
            BEST_PATHS_OEM.append((number_of_generations, pop[0][1]))
        
        counter += 1
    # return population

def create_list(popu):
    list_a = [x[1] for x in popu]
    return list_a

def best_worst_etc(paths):

    real_paths = []
    for elem in paths:
        real_paths.append(elem[1])
        # print(elem[1])
    # errors = []
    # for elem in paths:
    #     errors.append((elem[0], elem[1] - DISTANCE))
    
    # plt.plot(*zip(*errors))
    # plt.show()

    sorted(real_paths)
    best = min(real_paths)
    worst = max(real_paths)
    mean_of_runs = round(statistics.mean(real_paths))
    standard_deviation = round(statistics.stdev(real_paths))
    print("Best run: " + str(best) + "km.")
    print("Worst run: " + str(worst) + "km.")
    print("Mean of runs: " + str(mean_of_runs) + "km.")
    print("Standard deviation: : " + str(standard_deviation) + "km.")

def get_all(mydata, key):
    # creates a list of best distances based on the number of generation
    my_list = [item[1] for item in mydata if item[0] == key]
    return my_list

def get_error(paths):

    if PROBLEM == "gr17":
        real_dist = 2085
    if PROBLEM == "fri26":
        real_dist = 937
    if PROBLEM == "att48":
        real_dist = 33523

    error_each_path = []

    for elem in paths:
        err = abs(elem - real_dist) / real_dist * 100
        error_each_path.append(err)

    # fri26 937
    average_error = round(statistics.mean(error_each_path), 2)

    # gr17 2085

    # att48 33523
    return average_error, error_each_path

def refactor_data(path_arr):
    paths = sorted(path_arr, key=lambda x: x[0])

    paths_for_error = [i[1] for i in paths if i[0] == GENERATIONS]

    error, all_error = get_error(paths_for_error)
    
    new_paths = [(k, list(list(zip(*g))[1])) for k, g in groupby(paths, itemgetter(0))]
    even_newer_paths = []
    for elem in new_paths:
        even_newer_paths.append((elem[0], statistics.mean(elem[1])))

    return even_newer_paths, error, all_error

def get_time_data(arr):
    arr_mean = round(statistics.mean(arr))
    arr_max = max(arr)
    arr_min = min(arr)

    return arr_mean, arr_max, arr_min

def plot_error(rsm, iem, idm, oem):
    x_ind = numpy.arange(4)
    x_axis = ("RSM", "IEM", "IDM", "OEM")
    err_arr = [rsm, iem, idm, oem]

    plt.xticks(2 * x_ind, x_axis)

    err_bar = plt.bar(2 * x_ind, err_arr, width=0.4, color=["blue", "orange", "green", "red"], zorder=3)
    y_bottom, y_top = plt.ylim()
    plt.ylim(0, y_top + 20)
    for bar in err_bar:
        height = bar.get_height()
        plt.annotate('{}%'.format(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=8, rotation=0)

    plt.title("Average error percentage for each operator for {}".format(PROBLEM))
    plt.ylabel("Error percent [%]")
    plt.grid(True, zorder=0)
    plt.show()


def plot_times():
    x_ind = numpy.arange(4)
    x_axis = ("RSM", "IEM", "IDM", "OEM")
    rsm_times = [elem[1] for elem in TIMES_RSM]
    rsm_mean, rsm_worst, rsm_best = get_time_data(rsm_times)
    
    iem_times = [elem[1] for elem in TIMES_IEM]
    iem_mean, iem_worst, iem_best = get_time_data(iem_times)

    idm_times = [elem[1] for elem in TIMES_IDM]
    idm_mean, idm_worst, idm_best = get_time_data(idm_times)

    oem_times = [elem[1] for elem in TIMES_OEM]
    oem_mean, oem_worst, oem_best = get_time_data(oem_times)

    bests = [rsm_best, iem_best, idm_best, oem_best]
    worsts = [rsm_worst, iem_worst, idm_worst, oem_worst]
    means = [rsm_mean, iem_mean, idm_mean, oem_mean]

    best_bar = plt.bar(2 * x_ind - 0.4, bests, width=0.4, color="g", label="best", zorder=3)
    worst_bar = plt.bar(2 * x_ind, worsts, width=0.4, color="r", label="worst", zorder=3)
    mean_bar = plt.bar(2* x_ind + 0.4, means,  width=0.4, color="b", label="average", zorder=3)
    plt.xticks(2 * x_ind, x_axis)

    lim_bottom, lim_top = plt.ylim()
    
    if PROBLEM == "gr17":
        rot = 0
        plt.ylim(0, lim_top + 200)
    if PROBLEM == "fri26":
        rot = 55
        plt.ylim(0, lim_top + 400)
    if PROBLEM == "att48":
        plt.ylim(0, lim_top + 700)
        rot = 85

    for bar in best_bar:
        height = bar.get_height()
        plt.annotate('{}'.format(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height  + 0.5),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=8, rotation=rot)

    for bar in worst_bar:
        height = bar.get_height()
        plt.annotate('{}'.format(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height  + 0.5),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=8, rotation=rot)

    for bar in mean_bar:
        height = bar.get_height()
        plt.annotate('{}'.format(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height  + 0.5),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=8, rotation=rot)

    plt.title("GA runtimes over {} executions of the algorithm for {}".format(ITERATIONS, PROBLEM))
    plt.legend(loc="upper right")
    plt.ylabel("Time in miliseconds [ms]")
    plt.grid(True, zorder=0)
    plt.show()

def export_time():
    rsm = [x[1] for x in TIMES_RSM]
    iem = [x[1] for x in TIMES_IEM]
    idm = [x[1] for x in TIMES_IDM]
    oem = [x[1] for x in TIMES_OEM]
    df = pd.DataFrame({"RSM": rsm, "IEM": iem, "IDM": idm, "OEM": oem})
    path_final = "results//" + PROBLEM + ".xlsx"
    sheet = PROBLEM
    df.to_excel(path_final, sheet_name=sheet, index=False)

def export_error(rsm, iem, idm, oem):
    df = pd.DataFrame({"RSM": rsm, "IEM": iem, "IDM": idm, "OEM": oem})
    path_final = "results//" + PROBLEM + "_error.xlsx"
    sheet = PROBLEM + "_error"
    df.to_excel(path_final, sheet_name=sheet, index=False)

def genetic_algorithm():
    path_distances_gr17 = []

    population = init_population()
    for i in range(0, ITERATIONS):
        start_time = time.time()
        init_algorithm(population, "reverse")
        elapsed = time.time() - start_time
        TIMES_RSM.append((i + 1, round(elapsed * 1000, 2)))
    for i in range(0, ITERATIONS):  
        start_time = time.time()
        init_algorithm(population, "exchange")
        elapsed = time.time() - start_time
        TIMES_IEM.append((i + 1, round(elapsed * 1000, 2)))
    for i in range(0, ITERATIONS):
        start_time = time.time()
        init_algorithm(population, "displacement")
        elapsed = time.time() - start_time
        TIMES_IDM.append((i + 1, round(elapsed * 1000, 2)))
    for i in range(0, ITERATIONS):
        start_time = time.time()
        init_algorithm(population, "oddeven")
        elapsed = time.time() - start_time
        TIMES_OEM.append((i + 1, round(elapsed * 1000, 2)))

    
    rsm_plotting_data, rsm_err, rsm_all_err = refactor_data(BEST_PATHS_RSM)
    iem_plotting_data, iem_err, iem_all_err = refactor_data(BEST_PATHS_IEM)
    idm_plotting_data, idm_err, idm_all_err = refactor_data(BEST_PATHS_IDM)
    oem_plotting_data, oem_err, oem_all_err = refactor_data(BEST_PATHS_OEM)
    
    print("=== RSM ===")
    best_worst_etc(rsm_plotting_data)
    print("=== IEM ===")
    best_worst_etc(iem_plotting_data)
    print("=== IDM ===")
    best_worst_etc(idm_plotting_data)
    print("=== OEM ===")
    best_worst_etc(oem_plotting_data)

    plot_times()
    # plot_error(rsm_err, iem_err, idm_err, oem_err)
    export_time()
    export_error(rsm_all_err, iem_all_err, idm_all_err, oem_all_err)


    plt.plot(*zip(*rsm_plotting_data), label="RSM", color="blue")
    plt.plot(*zip(*iem_plotting_data), label="IEM", color="orange")
    plt.plot(*zip(*idm_plotting_data), label="IDM", color="green")
    plt.plot(*zip(*oem_plotting_data), label="OEM", color="red")
    plt.xlabel('Generations')
    plt.ylabel('Distance')
    plt.legend()
    plt.title("{} problem".format(PROBLEM))
    plt.grid(True)
    # plt.plot(*zip(*BEST_PATHS_OEM))
    plt.show()

    plot_error(rsm_err, iem_err, idm_err, oem_err)

    print()

