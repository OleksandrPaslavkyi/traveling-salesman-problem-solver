import random
import matplotlib as pyplot
import matplotlib.pyplot as plt
import numpy as np

##PARCER###

with open("berlin52.tsp", "r") as f:
    for _ in range(6):
        f.readline()

    data = []
    for line in f:
        line = line.strip()
        if line == 'EOF':
            break
        data.append(line)
data2 = [line.split() for line in data]
data3 = [[int(entry[0]), float(entry[1]), float(entry[2])] for entry in data2]

print("Raw coordinate lines (data):")
print(data)
print("\nSplit lines (data2):")
print(data2)
print("\nParsed coordinates (data3):")
print(data3)



def get_distance_between_two_dots(p1, p2):
    a = (((data3[p2 - 1][1] - data3[p1 - 1][1]) ** 2) + ((data3[p2 - 1][2] - data3[p1 - 1][2]) ** 2)) ** 0.5
    return a

list_of_cities = []
for i in range(len(data3)):
    list_of_cities.append(data3[i][0])
print(list_of_cities)

def random_itinerary_for_all_cities():
    total_distance = 0
    random.shuffle(list_of_cities)
    #print(list_of_cities)
    for i5 in range(len(list_of_cities)):
        if i5+2 > len(list_of_cities):
            break
        total_distance += get_distance_between_two_dots(list_of_cities[i5], list_of_cities[i5+1])
    #print(total_distance)
    return total_distance

random_solution_results = []
for i in range(10000):
    random_solution = random_itinerary_for_all_cities()
    random_solution_results.append(random_solution)

print(min(random_solution_results))


def info():
    total_distance = 0
    print(str(list_of_cities[0]) + " - " + "0")
    for info in range(len(list_of_cities)):
        if info+2 > len(list_of_cities):
            break
        total_distance += get_distance_between_two_dots(list_of_cities[info], list_of_cities[info+1])
        print(str(list_of_cities[info+1]) + " - " + str(total_distance))
    return total_distance


#info()


def greedy_solution(cities, start_point):
    current_city = start_point
    tour = [current_city]
    remaining_cities = [city for city in cities if city[0] != start_point]

    while remaining_cities:
        distances = [get_distance_between_two_dots(current_city, city[0]) for city in remaining_cities]
        nearest_city_index = distances.index(min(distances))
        nearest_city = remaining_cities[nearest_city_index]
        tour.append(nearest_city[0])
        current_city = nearest_city[0]
        remaining_cities.pop(nearest_city_index)

    total_distance = sum(get_distance_between_two_dots(tour[i], tour[i + 1]) for i in range(len(tour) - 1))
    total_distance += get_distance_between_two_dots(tour[-1], tour[0])
    tour.append(total_distance)

    return tour

def run_greedy_for_all_starting_cities(data):
    best_distance = float('inf')
    best_starting_point = None

    for starting_point in range(1, len(data) + 1):
        greedy_result = greedy_solution(data, starting_point)
        distance = greedy_result[-1]

        print(f"Starting City: {starting_point}, Distance: {distance}")

        if distance < best_distance:
            best_distance = distance
            best_starting_point = starting_point

    print("\nBest Starting Point:", best_starting_point)
    print("Best Distance:", best_distance)


run_greedy_for_all_starting_cities(data3)

def greedy_solution_statistics(data):
    results = []

    for starting_point in range(1, len(data) + 1):
        tour = greedy_solution(data, starting_point)
        distance = tour[-1]
        results.append(distance)

    return results

greedy_results = greedy_solution_statistics(data3)
def print_statistics_for_greedy(name, results):
    print(f"\nStatistics for {name}:")
    print("Mean:", np.mean(results))
    print("Standard Deviation:", np.std(results))
    print("Variance:", np.var(results))

print_statistics_for_greedy("Greedy Solution", greedy_results)


population = []

#list_of_cities_2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

def hundred_of_solution_using_fitness():

    for solution_i in range(len(data3)):
        list_of_cities_2 = list_of_cities.copy()
        random.shuffle(list_of_cities_2)
        total_distance_1 = 0
        for j1 in range(len(list_of_cities_2)):
            if j1+2 > len(list_of_cities_2):
                break
            #print(j1)
            total_distance_1 += (((data3[list_of_cities_2[j1+1] - 1][1] - data3[list_of_cities_2[j1] - 1][1]) ** 2) + ((data3[list_of_cities_2[j1+1] - 1][2] - data3[list_of_cities_2[j1] - 1][2]) ** 2)) ** 0.5
        total_distance_1 += (((data3[list_of_cities_2[-1] - 1][1] - data3[list_of_cities_2[0] - 1][1]) ** 2) + ((data3[list_of_cities_2[-1] -1][2] - data3[list_of_cities_2[0] - 1][2]) ** 2)) ** 0.5
        list_of_cities_2.append(total_distance_1)
        population.append(list_of_cities_2)
    #print(population)
    return population

hundred_of_solution_using_fitness()

results = []

def show_information_about_population(population):
    #for info in range(len(population)):
     #   results.append(population[info][-1])

    print("\n" + "Minimal distance is - " + str(min(population)))
    print("\n" + "Maximum distance is - " + str(max(population)))
    print("\n" + "Median distance is - " + str(sum(population)/len(population)))
    return results


def calculate_distance(individual):
    return individual[-1]


def tournament_selection_single(population, tournament_size):
    tournament_candidates = random.sample(population, tournament_size)
    winner = min(tournament_candidates, key=calculate_distance)
    return winner[:-1]


def cycle_crossover(parent1, parent2):
    size = len(parent1)
    child = [-1] * size
    start_index = random.randint(0, size - 1)

    while child[start_index] == -1:
        child[start_index] = parent1[start_index]
        index = parent2.index(parent1[start_index])
        start_index = index

    for i in range(size):
        if child[i] == -1:
            child[i] = parent2[i]

    return child


list_of_every_best_result_tournament_5_mutation_10 = []
list_of_every_best_result_tournament_10_mutation_10 = []
list_of_every_best_result_tournament_20_mutation_10 = []
list_of_every_best_result_tournament_30_mutation_10 = []

list_of_every_best_result_tournament_5_mutation_20 = []
list_of_every_best_result_tournament_10_mutation_20 = []
list_of_every_best_result_tournament_20_mutation_20 = []
list_of_every_best_result_tournament_30_mutation_20 = []

list_of_every_best_result_tournament_5_mutation_5 = []
list_of_every_best_result_tournament_10_mutation_5 = []
list_of_every_best_result_tournament_20_mutation_5 = []
list_of_every_best_result_tournament_30_mutation_5 = []



def mutation(child, mutation_probability):
    if random.random() < mutation_probability:
        start_index = random.randint(0, len(child) - 1)
        end_index = random.randint(start_index + 1, len(child))
        child[start_index:end_index] = reversed(child[start_index:end_index])
    return child
def create_epoch(previous_epoch, tournament_size, mutation_probability):
    new_epoch = []

    for i in range(len(data3)):
        parent1 = tournament_selection_single(previous_epoch, tournament_size)
        parent2 = tournament_selection_single(previous_epoch, tournament_size)
        child = cycle_crossover(parent1, parent2)
        child = mutation(child, mutation_probability)
        total_distance = 0

        for distance in range(len(child) - 1):
            total_distance += get_distance_between_two_dots(child[distance], child[distance + 1])

        total_distance += get_distance_between_two_dots(child[-1], child[0])

        child.append(total_distance)
        new_epoch.append(child)

    best_results = []
    for j in range(len(new_epoch)):
        best_results.append(new_epoch[j][-1])

    if tournament_size == 10 and mutation_probability == 0.1:
        list_of_every_best_result_tournament_10_mutation_10.append(min(best_results))
    elif tournament_size == 20 and mutation_probability == 0.1:
        list_of_every_best_result_tournament_20_mutation_10.append(min(best_results))
    elif tournament_size == 30 and mutation_probability == 0.1:
        list_of_every_best_result_tournament_30_mutation_10.append(min(best_results))
    elif tournament_size == 5 and mutation_probability == 0.1:
        list_of_every_best_result_tournament_5_mutation_10.append(min(best_results))
    elif tournament_size == 10 and mutation_probability == 0.2:
        list_of_every_best_result_tournament_10_mutation_20.append(min(best_results))
    elif tournament_size == 20 and mutation_probability == 0.2:
        list_of_every_best_result_tournament_20_mutation_20.append(min(best_results))
    elif tournament_size == 30 and mutation_probability == 0.2:
        list_of_every_best_result_tournament_30_mutation_20.append(min(best_results))
    elif tournament_size == 5 and mutation_probability == 0.2:
        list_of_every_best_result_tournament_5_mutation_20.append(min(best_results))
    elif tournament_size == 10 and mutation_probability == 0.05:
        list_of_every_best_result_tournament_10_mutation_5.append(min(best_results))
    elif tournament_size == 20 and mutation_probability == 0.05:
        list_of_every_best_result_tournament_20_mutation_5.append(min(best_results))
    elif tournament_size == 30 and mutation_probability == 0.05:
        list_of_every_best_result_tournament_30_mutation_5.append(min(best_results))
    elif tournament_size == 5 and mutation_probability == 0.05:
        list_of_every_best_result_tournament_5_mutation_5.append(min(best_results))


    return new_epoch


population_tournament_10_mutation_10 = hundred_of_solution_using_fitness()
population_tournament_20_mutation_10 = hundred_of_solution_using_fitness()
population_tournament_30_mutation_10 = hundred_of_solution_using_fitness()
population_tournament_5_mutation_10 = hundred_of_solution_using_fitness()
population_tournament_10_mutation_20 = hundred_of_solution_using_fitness()
population_tournament_20_mutation_20 = hundred_of_solution_using_fitness()
population_tournament_30_mutation_20 = hundred_of_solution_using_fitness()
population_tournament_5_mutation_20 = hundred_of_solution_using_fitness()
population_tournament_10_mutation_5 = hundred_of_solution_using_fitness()
population_tournament_20_mutation_5 = hundred_of_solution_using_fitness()
population_tournament_30_mutation_5 = hundred_of_solution_using_fitness()
population_tournament_5_mutation_5 = hundred_of_solution_using_fitness()

generations = 1500



for generation in range(generations):
    population_tournament_10_mutation_10 = create_epoch(population_tournament_10_mutation_10, 10, 0.1)
    population_tournament_20_mutation_10 = create_epoch(population_tournament_20_mutation_10, 20, 0.1)
    population_tournament_30_mutation_10 = create_epoch(population_tournament_30_mutation_10, 30, 0.1)
    population_tournament_5_mutation_10 = create_epoch(population_tournament_5_mutation_10, 5, 0.1)
    population_tournament_10_mutation_20 = create_epoch(population_tournament_10_mutation_20, 10, 0.2)
    population_tournament_20_mutation_20 = create_epoch(population_tournament_20_mutation_20, 20, 0.2)
    population_tournament_30_mutation_20 = create_epoch(population_tournament_30_mutation_20, 30, 0.2)
    population_tournament_5_mutation_20 = create_epoch(population_tournament_5_mutation_20, 5, 0.2)
    population_tournament_10_mutation_5 = create_epoch(population_tournament_10_mutation_5, 10, 0.05)
    population_tournament_20_mutation_5 = create_epoch(population_tournament_20_mutation_5, 20, 0.05)
    population_tournament_30_mutation_5 = create_epoch(population_tournament_30_mutation_5, 30, 0.05)
    population_tournament_5_mutation_5 = create_epoch(population_tournament_5_mutation_5, 5, 0.05)





results_variants = [
    ("Tournament Size = 10, Mutation = 10%", list_of_every_best_result_tournament_10_mutation_10),
    ("Tournament Size = 20 Mutation = 10%", list_of_every_best_result_tournament_20_mutation_10),
    ("Tournament Size = 30 Mutation = 10%", list_of_every_best_result_tournament_30_mutation_10),
    ("Tournament Size = 5 Mutation = 10%", list_of_every_best_result_tournament_5_mutation_10),
    ("Tournament Size = 10 Mutation = 20%", list_of_every_best_result_tournament_10_mutation_20),
    ("Tournament Size = 20 Mutation = 20%", list_of_every_best_result_tournament_20_mutation_20),
    ("Tournament Size = 30 Mutation = 20%", list_of_every_best_result_tournament_30_mutation_20),
    ("Tournament Size = 5 Mutation = 20%", list_of_every_best_result_tournament_5_mutation_20),
    ("Tournament Size = 10 Mutation = 5%", list_of_every_best_result_tournament_10_mutation_5),
    ("Tournament Size = 20 Mutation = 5%", list_of_every_best_result_tournament_20_mutation_5),
    ("Tournament Size = 30 Mutation = 5%", list_of_every_best_result_tournament_30_mutation_5),
    ("Tournament Size = 5 Mutation = 5%", list_of_every_best_result_tournament_5_mutation_5),
]


overall_best_solution = float('inf')
overall_best_variant = None

for name, variant in results_variants:
    min_value = min(variant)
    if min_value < overall_best_solution:
        overall_best_solution = min_value
        overall_best_variant = name

print("\nBest Solution:", overall_best_solution)
print(overall_best_variant)

def calculate_population_statistics(population):
    results = [individual[-1] for individual in population]
    return results


results_tournament_10_mutation_10 = calculate_population_statistics(population_tournament_10_mutation_10)
results_tournament_20_mutation_10 = calculate_population_statistics(population_tournament_20_mutation_10)
results_tournament_30_mutation_10 = calculate_population_statistics(population_tournament_30_mutation_10)
results_tournament_5_mutation_10 = calculate_population_statistics(population_tournament_5_mutation_10)
results_tournament_10_mutation_20 = calculate_population_statistics(population_tournament_10_mutation_20)
results_tournament_20_mutation_20 = calculate_population_statistics(population_tournament_20_mutation_20)
results_tournament_30_mutation_20 = calculate_population_statistics(population_tournament_30_mutation_20)
results_tournament_5_mutation_20 = calculate_population_statistics(population_tournament_5_mutation_20)
results_tournament_10_mutation_5 = calculate_population_statistics(population_tournament_10_mutation_5)
results_tournament_20_mutation_5 = calculate_population_statistics(population_tournament_20_mutation_5)
results_tournament_30_mutation_5 = calculate_population_statistics(population_tournament_30_mutation_5)
results_tournament_5_mutation_5 = calculate_population_statistics(population_tournament_5_mutation_5)


def print_statistics(name, results):
    print(f"\nStatistics for {name}:")
    print("Mean:", np.mean(results))
    print("Standard Deviation:", np.std(results))
    print("Variance:", np.var(results))


print_statistics("Tournament Size = 10, Mutation = 10%", results_tournament_10_mutation_10)
print_statistics("Tournament Size = 20 Mutation = 10%", results_tournament_20_mutation_10)
print_statistics("Tournament Size = 30 Mutation = 10%", results_tournament_30_mutation_10)
print_statistics("Tournament Size = 5 Mutation = 10%", results_tournament_5_mutation_10)
print_statistics("Tournament Size = 10 Mutation = 20%", results_tournament_10_mutation_20)
print_statistics("Tournament Size = 20 Mutation = 20%", results_tournament_20_mutation_20)
print_statistics("Tournament Size = 30 Mutation = 20%", results_tournament_30_mutation_20)
print_statistics("Tournament Size = 5 Mutation = 20%", results_tournament_5_mutation_20)
print_statistics("Tournament Size = 10, Mutation = 5%", results_tournament_10_mutation_5)
print_statistics("Tournament Size = 20 Mutation = 5%", results_tournament_20_mutation_5)
print_statistics("Tournament Size = 30 Mutation = 5%", results_tournament_30_mutation_5)
print_statistics("Tournament Size = 5 Mutation = 5%", results_tournament_5_mutation_5)




plt.subplot(3, 1, 1)

xpoints = list(range(1, generations + 1))
plt.title("Improvement graph")
plt.plot(xpoints, list_of_every_best_result_tournament_10_mutation_10, label='Tournament Size = 10, Mutation = 10%')
plt.plot(xpoints, list_of_every_best_result_tournament_20_mutation_10, label='Tournament Size = 20 Mutation = 10%')
plt.plot(xpoints, list_of_every_best_result_tournament_30_mutation_10, label='Tournament Size = 30 Mutation = 10%')
plt.plot(xpoints, list_of_every_best_result_tournament_5_mutation_10, label='Tournament Size = 5 Mutation = 10%')
plt.legend()

plt.subplot(3, 1, 2)

plt.plot(xpoints, list_of_every_best_result_tournament_10_mutation_20, label='Tournament Size = 10 Mutation = 20%')
plt.plot(xpoints, list_of_every_best_result_tournament_20_mutation_20, label='Tournament Size = 20 Mutation = 20%')
plt.plot(xpoints, list_of_every_best_result_tournament_30_mutation_20, label='Tournament Size = 30 Mutation = 20%')
plt.plot(xpoints, list_of_every_best_result_tournament_5_mutation_20, label='Tournament Size = 5 Mutation = 20%')
plt.legend()

plt.subplot(3, 1, 3)

plt.plot(xpoints, list_of_every_best_result_tournament_10_mutation_5, label='Tournament Size = 10 Mutation = 5%')
plt.plot(xpoints, list_of_every_best_result_tournament_20_mutation_5, label='Tournament Size = 20 Mutation = 5%')
plt.plot(xpoints, list_of_every_best_result_tournament_30_mutation_5, label='Tournament Size = 30 Mutation = 5%')
plt.plot(xpoints, list_of_every_best_result_tournament_5_mutation_5, label='Tournament Size = 5 Mutation = 5%')
plt.legend()
plt.show()



