from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Iterable

from data_loader import DietData, Food, User


NUTRIENT_IDS = {
    "energy": 5,
    "protein": 15,
    "carbohydrate": 8,
    "fiber": 4,
    "sodium": 17,
}

BREAKFAST_GROUPS = set(range(0, 15))
LUNCH_DINNER_GROUPS = set(range(15, 29))
MEAT_GROUPS = {2, 3, 15, 23, 28}
MEAT_WORDS = (
    "ET",
    "KIYMA",
    "KÖFTE",
    "TAVUK",
    "PİLİÇ",
    "HİNDİ",
    "BALIK",
    "SALAM",
    "SUCUK",
)


@dataclass
class Chromosome:
    breakfast: list[int]
    lunch_dinner: list[int]


@dataclass
class Menu:
    breakfast_ids: list[int]
    lunch_dinner_ids: list[int]
    nutrients: dict[int, float]
    objectives: dict[str, float]
    penalty: float
    distinct_groups: int
    violations: int
    vegetarian_forbidden_count: int
    fitness: tuple[float, ...] = field(default_factory=tuple)

    @property
    def all_food_ids(self) -> list[int]:
        return self.breakfast_ids + self.lunch_dinner_ids


class DietProblem:
    def __init__(
        self,
        data: DietData,
        user: User,
        objective_names: tuple[str, ...] = ("preference", "cost", "time"),
        diversity: bool = True,
        penalty_weight: float = 1000.0,
        diversity_alpha: float = 0.25,
        vegetarian_override: bool | None = None,
    ) -> None:
        self.data = data
        self.user = user
        self.objective_names = objective_names
        self.diversity = diversity
        self.penalty_weight = penalty_weight
        self.diversity_alpha = diversity_alpha
        self.nutrient_ids = list(NUTRIENT_IDS.values())
        self.breakfast_nutrients = [NUTRIENT_IDS["energy"], NUTRIENT_IDS["protein"]]
        self.dri = self._dri_for_user(user)
        self.preferences = self._preferences_for_user(user)
        self.is_vegetarian = self._detect_vegetarian_user() if vegetarian_override is None else vegetarian_override
        self.breakfast_pool = self._pool(BREAKFAST_GROUPS)
        self.lunch_dinner_pool = self._pool(LUNCH_DINNER_GROUPS)
        if not self.breakfast_pool or not self.lunch_dinner_pool:
            raise ValueError("Could not build breakfast and lunch+dinner pools from food groups.")

    def random_chromosome(self, rng: random.Random) -> Chromosome:
        breakfast = self.breakfast_pool[:]
        lunch_dinner = self.lunch_dinner_pool[:]
        rng.shuffle(breakfast)
        rng.shuffle(lunch_dinner)
        return Chromosome(breakfast, lunch_dinner)

    def evaluate(self, chromosome: Chromosome) -> Menu:
        totals = {nid: 0.0 for nid in self.nutrient_ids}
        breakfast = self._decode_part(
            chromosome.breakfast,
            totals,
            self.breakfast_nutrients,
            lower_scale=0.35 * 0.90,
            upper_scale=0.35 * 1.15,
        )
        lunch_dinner = self._decode_part(
            chromosome.lunch_dinner,
            totals,
            self.nutrient_ids,
            lower_scale=0.90,
            upper_scale=1.15,
        )
        selected = breakfast + lunch_dinner
        objectives = self._objectives(selected)
        penalty, violations = self._nutrient_penalty(totals)
        distinct_groups = len({self.data.foods[food_id].food_group_id for food_id in selected})
        forbidden = sum(1 for food_id in selected if self._is_forbidden(food_id))
        total_penalty = penalty
        if self.diversity:
            total_penalty += self.diversity_alpha * (1.0 / max(1, distinct_groups))
        total_penalty += 10.0 * forbidden

        fitness: list[float] = []
        for name in self.objective_names:
            value = objectives[name]
            if name == "preference":
                fitness.append(-value + self.penalty_weight * total_penalty)
            else:
                fitness.append(value + self.penalty_weight * total_penalty)

        return Menu(
            breakfast_ids=breakfast,
            lunch_dinner_ids=lunch_dinner,
            nutrients=totals.copy(),
            objectives=objectives,
            penalty=total_penalty,
            distinct_groups=distinct_groups,
            violations=violations,
            vegetarian_forbidden_count=forbidden,
            fitness=tuple(fitness),
        )

    def food_names(self, food_ids: Iterable[int]) -> list[str]:
        return [self.data.foods[food_id].name for food_id in food_ids]

    def dri_bounds(self, nutrient_id: int) -> tuple[float, float]:
        return self.dri[nutrient_id]

    def _decode_part(
        self,
        genes: list[int],
        totals: dict[int, float],
        check_nutrients: list[int],
        lower_scale: float,
        upper_scale: float,
    ) -> list[int]:
        selected: list[int] = []
        for food_id in genes:
            food = self.data.foods[food_id]
            if self._is_forbidden(food_id):
                continue
            if any(
                totals[nid] + food.nutrients.get(nid, 0.0) > self.dri[nid][1] * upper_scale
                for nid in check_nutrients
            ):
                continue
            selected.append(food_id)
            for nid in self.nutrient_ids:
                totals[nid] += food.nutrients.get(nid, 0.0)
            if all(totals[nid] >= self.dri[nid][0] * lower_scale for nid in check_nutrients):
                break
        return selected

    def _objectives(self, selected: list[int]) -> dict[str, float]:
        foods = [self.data.foods[food_id] for food_id in selected]
        return {
            "preference": sum(self.preferences.get(food.id, 0.0) for food in foods),
            "cost": sum(food.cost for food in foods),
            "time": sum(food.preparing_time + food.cooking_time for food in foods),
            "co2": sum(food.co2 for food in foods),
        }

    def _nutrient_penalty(self, totals: dict[int, float]) -> tuple[float, int]:
        penalty = 0.0
        violations = 0
        for nid in self.nutrient_ids:
            low, high = self.dri[nid]
            span = max(high - low, 1e-9)
            low_violation = max(0.0, low - totals[nid]) / span
            high_violation = max(0.0, totals[nid] - high) / span
            if low_violation > 0 or high_violation > 0:
                violations += 1
            penalty += 0.7 * low_violation + 0.3 * high_violation
        return penalty, violations

    def _dri_for_user(self, user: User) -> dict[int, tuple[float, float]]:
        gender = user.gender.lower()
        if gender not in {"male", "female", "child", "pregnant"}:
            gender = "child" if user.age < 9 else "male"
        if user.age < 9:
            gender = "child"
        bounds: dict[int, tuple[float, float]] = {}
        for nid in self.nutrient_ids:
            matches = [
                row
                for row in self.data.dri_rows
                if row["nutrient_id"] == nid
                and row["gender"] == gender
                and row["low_age"] <= user.age <= row["up_age"]
            ]
            if not matches and gender == "child":
                matches = [
                    row
                    for row in self.data.dri_rows
                    if row["nutrient_id"] == nid
                    and row["low_age"] <= user.age <= row["up_age"]
                ]
            if not matches:
                raise ValueError(f"No DRI row for nutrient {nid}, age={user.age}, gender={gender}")
            row = matches[0]
            bounds[nid] = (float(row["RLL"]), float(row["RUL"]))
        return bounds

    def _preferences_for_user(self, user: User) -> dict[int, float]:
        explicit = self.data.user_preferences.get(user.id, {})
        if explicit:
            return {food_id: explicit.get(food_id, 0.0) for food_id in self.data.foods}

        column = "preference2" if self._food_column_looks_vegetarian("preference2") else "preference"
        return {
            food_id: getattr(food, column)
            for food_id, food in self.data.foods.items()
        }

    def _detect_vegetarian_user(self) -> bool:
        explicit = self.data.user_preferences.get(self.user.id, {})
        if explicit:
            meat_scores = [
                score
                for food_id, score in explicit.items()
                if food_id in self.data.foods and self._looks_meat_based(self.data.foods[food_id])
            ]
            if meat_scores:
                return min(meat_scores) <= -1
        return self._food_column_looks_vegetarian("preference2")

    def _food_column_looks_vegetarian(self, column: str) -> bool:
        meat_scores = [
            getattr(food, column)
            for food in self.data.foods.values()
            if self._looks_meat_based(food)
        ]
        return bool(meat_scores) and sum(1 for score in meat_scores if score <= -1) >= 5

    def _pool(self, groups: set[int]) -> list[int]:
        return [
            food.id
            for food in self.data.foods.values()
            if food.food_group_id in groups
        ]

    def _is_forbidden(self, food_id: int) -> bool:
        score = self.preferences.get(food_id, 0.0)
        if score <= -1:
            return True
        return self.is_vegetarian and self._looks_meat_based(self.data.foods[food_id])

    @staticmethod
    def _looks_meat_based(food: Food) -> bool:
        upper_name = food.name.upper()
        return food.food_group_id in MEAT_GROUPS or any(word in upper_name for word in MEAT_WORDS)


def dominates(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def non_dominated_indices(fitnesses: list[tuple[float, ...]]) -> list[int]:
    result: list[int] = []
    for i, fit in enumerate(fitnesses):
        if not any(dominates(other, fit) for j, other in enumerate(fitnesses) if i != j):
            result.append(i)
    return result


def hypervolume_2d_or_3d(points: list[tuple[float, ...]], reference: tuple[float, ...]) -> float:
    if not points:
        return 0.0
    dims = len(reference)
    clipped = [tuple(min(p[i], reference[i]) for i in range(dims)) for p in points]
    nd = [clipped[i] for i in non_dominated_indices(clipped)]
    if dims == 2:
        return _hv2(nd, reference)
    if dims == 3:
        return _hv3(nd, reference)
    return 0.0


def _hv2(points: list[tuple[float, ...]], reference: tuple[float, ...]) -> float:
    total = 0.0
    prev_y = reference[1]
    for x, y in sorted(points):
        width = max(0.0, reference[0] - x)
        height = max(0.0, prev_y - y)
        total += width * height
        prev_y = min(prev_y, y)
    return total


def _hv3(points: list[tuple[float, ...]], reference: tuple[float, ...]) -> float:
    xs = sorted({p[0] for p in points} | {reference[0]})
    total = 0.0
    for left, right in zip(xs, xs[1:]):
        width = right - left
        active = [(p[1], p[2]) for p in points if p[0] <= left]
        if width > 0 and active:
            total += width * _hv2(active, (reference[1], reference[2]))
    return total


def finite_or_zero(value: float) -> float:
    return value if math.isfinite(value) else 0.0
