from __future__ import annotations

import random
from dataclasses import dataclass

from modp import Chromosome, DietProblem, Menu, dominates, non_dominated_indices


@dataclass
class Individual:
    chromosome: Chromosome
    menu: Menu
    rank: int = 0
    crowding: float = 0.0
    strength: int = 0
    raw_fitness: float = 0.0
    density: float = 0.0


@dataclass
class RunResult:
    algorithm: str
    population: list[Individual]
    pareto: list[Individual]
    hypervolume_history: list[float]
    front_history: list[list[tuple[float, ...]]]


def run_nsga2(
    problem: DietProblem,
    generations: int,
    population_size: int,
    seed: int,
    crossover_rate: float = 0.9,
) -> RunResult:
    rng = random.Random(seed)
    population = [_new_individual(problem, rng) for _ in range(population_size)]
    hv_history: list[float] = []
    front_history: list[list[tuple[float, ...]]] = []

    for _ in range(generations):
        fronts = _assign_rank_and_crowding(population)
        offspring: list[Individual] = []
        while len(offspring) < population_size:
            p1 = _tournament_nsga(population, rng)
            p2 = _tournament_nsga(population, rng)
            c1, c2 = _reproduce(problem, p1.chromosome, p2.chromosome, rng, crossover_rate)
            offspring.append(Individual(c1, problem.evaluate(c1)))
            if len(offspring) < population_size:
                offspring.append(Individual(c2, problem.evaluate(c2)))
        combined = population + offspring
        fronts = _assign_rank_and_crowding(combined)
        population = _survivors_nsga(fronts, population_size)
        front_history.append([ind.menu.fitness for ind in _pareto(population)])

    pareto = _pareto(population)
    return RunResult("NSGA-II", population, pareto, hv_history, front_history)


def run_spea2(
    problem: DietProblem,
    generations: int,
    population_size: int,
    seed: int,
    archive_size: int | None = None,
    crossover_rate: float = 0.9,
) -> RunResult:
    rng = random.Random(seed)
    archive_size = archive_size or population_size
    population = [_new_individual(problem, rng) for _ in range(population_size)]
    archive: list[Individual] = []
    hv_history: list[float] = []
    front_history: list[list[tuple[float, ...]]] = []

    for _ in range(generations):
        union = population + archive
        _assign_spea2_fitness(union)
        archive = [ind for ind in union if ind.raw_fitness < 1.0]
        if len(archive) < archive_size:
            archive.extend(sorted(union, key=lambda ind: ind.raw_fitness + ind.density)[: archive_size - len(archive)])
        elif len(archive) > archive_size:
            archive = _truncate_archive(archive, archive_size)

        mating_pool = archive or population
        population = []
        while len(population) < population_size:
            p1 = _tournament_spea(mating_pool, rng)
            p2 = _tournament_spea(mating_pool, rng)
            c1, c2 = _reproduce(problem, p1.chromosome, p2.chromosome, rng, crossover_rate)
            population.append(Individual(c1, problem.evaluate(c1)))
            if len(population) < population_size:
                population.append(Individual(c2, problem.evaluate(c2)))
        front_history.append([ind.menu.fitness for ind in _pareto(archive or population)])

    pareto = _pareto(archive or population)
    return RunResult("SPEA2", archive or population, pareto, hv_history, front_history)


def _new_individual(problem: DietProblem, rng: random.Random) -> Individual:
    chromosome = problem.random_chromosome(rng)
    return Individual(chromosome, problem.evaluate(chromosome))


def _reproduce(
    problem: DietProblem,
    parent1: Chromosome,
    parent2: Chromosome,
    rng: random.Random,
    crossover_rate: float,
) -> tuple[Chromosome, Chromosome]:
    if rng.random() < crossover_rate:
        b1, b2 = _order_crossover(parent1.breakfast, parent2.breakfast, rng)
        l1, l2 = _order_crossover(parent1.lunch_dinner, parent2.lunch_dinner, rng)
    else:
        b1, b2 = parent1.breakfast[:], parent2.breakfast[:]
        l1, l2 = parent1.lunch_dinner[:], parent2.lunch_dinner[:]
    _swap_mutation(b1, rng)
    _swap_mutation(b2, rng)
    _swap_mutation(l1, rng)
    _swap_mutation(l2, rng)
    return Chromosome(b1, l1), Chromosome(b2, l2)


def _order_crossover(a: list[int], b: list[int], rng: random.Random) -> tuple[list[int], list[int]]:
    if len(a) < 2:
        return a[:], b[:]
    i, j = sorted(rng.sample(range(len(a)), 2))
    return _ox_child(a, b, i, j), _ox_child(b, a, i, j)


def _ox_child(a: list[int], b: list[int], i: int, j: int) -> list[int]:
    child = [None] * len(a)
    child[i:j] = a[i:j]
    fill = [gene for gene in b if gene not in child]
    k = 0
    for idx in list(range(j, len(a))) + list(range(0, j)):
        if child[idx] is None:
            child[idx] = fill[k]
            k += 1
    return [int(gene) for gene in child]


def _swap_mutation(genes: list[int], rng: random.Random) -> None:
    probability = 1.0 / max(1, len(genes))
    for _ in range(len(genes)):
        if rng.random() < probability:
            i, j = rng.sample(range(len(genes)), 2)
            genes[i], genes[j] = genes[j], genes[i]


def _assign_rank_and_crowding(population: list[Individual]) -> list[list[Individual]]:
    fronts: list[list[Individual]] = []
    remaining = population[:]
    rank = 0
    while remaining:
        fitnesses = [ind.menu.fitness for ind in remaining]
        front_indices = set(non_dominated_indices(fitnesses))
        front = [ind for i, ind in enumerate(remaining) if i in front_indices]
        for ind in front:
            ind.rank = rank
        _assign_crowding(front)
        fronts.append(front)
        remaining = [ind for i, ind in enumerate(remaining) if i not in front_indices]
        rank += 1
    return fronts


def _assign_crowding(front: list[Individual]) -> None:
    if not front:
        return
    for ind in front:
        ind.crowding = 0.0
    dimensions = len(front[0].menu.fitness)
    for dim in range(dimensions):
        front.sort(key=lambda ind: ind.menu.fitness[dim])
        front[0].crowding = front[-1].crowding = float("inf")
        low = front[0].menu.fitness[dim]
        high = front[-1].menu.fitness[dim]
        if high == low:
            continue
        for i in range(1, len(front) - 1):
            front[i].crowding += (
                front[i + 1].menu.fitness[dim] - front[i - 1].menu.fitness[dim]
            ) / (high - low)


def _survivors_nsga(fronts: list[list[Individual]], population_size: int) -> list[Individual]:
    survivors: list[Individual] = []
    for front in fronts:
        if len(survivors) + len(front) <= population_size:
            survivors.extend(front)
        else:
            survivors.extend(sorted(front, key=lambda ind: ind.crowding, reverse=True)[: population_size - len(survivors)])
            break
    return survivors


def _tournament_nsga(population: list[Individual], rng: random.Random) -> Individual:
    a, b = rng.sample(population, 2)
    if (a.rank, -a.crowding) < (b.rank, -b.crowding):
        return a
    return b


def _assign_spea2_fitness(population: list[Individual]) -> None:
    n = len(population)
    for ind in population:
        ind.strength = 0
        ind.raw_fitness = 0.0
        ind.density = 0.0
    for i, a in enumerate(population):
        for j, b in enumerate(population):
            if i != j and dominates(a.menu.fitness, b.menu.fitness):
                a.strength += 1
    for i, a in enumerate(population):
        a.raw_fitness = sum(
            b.strength
            for j, b in enumerate(population)
            if i != j and dominates(b.menu.fitness, a.menu.fitness)
        )
        distances = sorted(_distance(a.menu.fitness, b.menu.fitness) for j, b in enumerate(population) if i != j)
        k = max(0, int(n ** 0.5) - 1)
        sigma = distances[k] if distances else 0.0
        a.density = 1.0 / (sigma + 2.0)


def _truncate_archive(archive: list[Individual], size: int) -> list[Individual]:
    archive = archive[:]
    while len(archive) > size:
        nearest = []
        for i, ind in enumerate(archive):
            distances = sorted(_distance(ind.menu.fitness, other.menu.fitness) for j, other in enumerate(archive) if i != j)
            nearest.append((distances, i))
        _, remove_i = min(nearest)
        archive.pop(remove_i)
    return archive


def _tournament_spea(population: list[Individual], rng: random.Random) -> Individual:
    a, b = rng.sample(population, 2)
    return a if (a.raw_fitness + a.density) <= (b.raw_fitness + b.density) else b


def _distance(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def _pareto(population: list[Individual]) -> list[Individual]:
    fitnesses = [ind.menu.fitness for ind in population]
    return [population[i] for i in non_dominated_indices(fitnesses)]

