from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Food:
    id: int
    name: str
    food_group_id: int
    portion: int
    cost: float
    preference: float
    preference2: float
    preparing_time: float
    cooking_time: float
    rating: float
    co2: float
    availability: int
    nutrients: dict[int, float]


@dataclass(frozen=True)
class User:
    id: int
    name: str
    surname: str
    username: str
    age: int
    gender: str
    height: int
    weight: int


@dataclass
class DietData:
    foods: dict[int, Food]
    food_groups: dict[int, str]
    nutrients: dict[int, str]
    dri_rows: list[dict[str, Any]]
    users: list[User]
    user_preferences: dict[int, dict[int, float]]


TABLE_COLUMNS = {
    "foods": [
        "id",
        "name",
        "foodGroupId",
        "caseStudy",
        "portion",
        "cost",
        "preference",
        "preference2",
        "preparingTime",
        "cookingTime",
        "rating",
        "co2",
        "availability",
        "picUrl",
        "nameRoots",
    ],
    "food_group": ["id", "name"],
    "food_nutrients": ["id", "foodId", "nutrientId", "quantity"],
    "nutrients": ["id", "name", "nGroupId", "unitId"],
    "dri": ["id", "nutrient_id", "low_age", "up_age", "gender", "RLL", "RUL"],
    "user": [
        "id",
        "name",
        "surname",
        "username",
        "password",
        "type",
        "date",
        "age",
        "gender",
        "height",
        "weight",
    ],
    "user_foods": ["id", "userId", "userName", "foodId", "foodName", "preference"],
}


def load_diet_data(sql_path: str | Path) -> DietData:
    text = Path(sql_path).read_text(encoding="utf-8")
    raw_tables = {table: _read_insert_rows(text, table) for table in TABLE_COLUMNS}

    food_groups = {int(row["id"]): str(row["name"]) for row in raw_tables["food_group"]}
    nutrients = {int(row["id"]): str(row["name"]) for row in raw_tables["nutrients"]}

    food_nutrients: dict[int, dict[int, float]] = {}
    for row in raw_tables["food_nutrients"]:
        food_nutrients.setdefault(int(row["foodId"]), {})[int(row["nutrientId"])] = float(row["quantity"])

    foods: dict[int, Food] = {}
    for row in raw_tables["foods"]:
        food_id = int(row["id"])
        foods[food_id] = Food(
            id=food_id,
            name=str(row["name"]),
            food_group_id=int(row["foodGroupId"]),
            portion=int(row["portion"]),
            cost=float(row["cost"]),
            preference=float(row["preference"]),
            preference2=float(row["preference2"]),
            preparing_time=float(row["preparingTime"]),
            cooking_time=float(row["cookingTime"] or 0),
            rating=float(row["rating"]),
            co2=float(row["co2"]),
            availability=int(row["availability"]),
            nutrients=food_nutrients.get(food_id, {}),
        )

    users = [
        User(
            id=int(row["id"]),
            name=str(row["name"] or ""),
            surname=str(row["surname"] or ""),
            username=str(row["username"]),
            age=int(row["age"] or 30),
            gender=str(row["gender"] or "adult"),
            height=int(row["height"] or 170),
            weight=int(row["weight"] or 70),
        )
        for row in raw_tables["user"]
    ]

    user_preferences: dict[int, dict[int, float]] = {}
    for row in raw_tables["user_foods"]:
        if row["preference"] is None or row["foodId"] is None:
            continue
        user_preferences.setdefault(int(row["userId"]), {})[int(row["foodId"])] = float(row["preference"])

    dri_rows = [
        {
            "nutrient_id": int(row["nutrient_id"]),
            "low_age": int(row["low_age"]),
            "up_age": int(row["up_age"]),
            "gender": str(row["gender"]).lower(),
            "RLL": float(row["RLL"]),
            "RUL": float(row["RUL"]),
        }
        for row in raw_tables["dri"]
    ]

    return DietData(
        foods=foods,
        food_groups=food_groups,
        nutrients=nutrients,
        dri_rows=dri_rows,
        users=users,
        user_preferences=user_preferences,
    )


def _read_insert_rows(sql_text: str, table: str) -> list[dict[str, Any]]:
    columns = TABLE_COLUMNS[table]
    pattern = re.compile(
        rf"INSERT INTO `{re.escape(table)}` \([^)]+\) VALUES\s*(.*?);",
        flags=re.DOTALL,
    )
    rows: list[dict[str, Any]] = []
    for match in pattern.finditer(sql_text):
        for values in _split_tuples(match.group(1)):
            parsed = _parse_tuple(values)
            rows.append(dict(zip(columns, parsed)))
    return rows


def _split_tuples(values_sql: str) -> list[str]:
    tuples: list[str] = []
    in_quote = False
    escape = False
    depth = 0
    start = -1
    for i, ch in enumerate(values_sql):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == "'":
            in_quote = not in_quote
            continue
        if in_quote:
            continue
        if ch == "(":
            if depth == 0:
                start = i + 1
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0 and start >= 0:
                tuples.append(values_sql[start:i])
    return tuples


def _parse_tuple(tuple_sql: str) -> list[Any]:
    normalized = re.sub(r"\bNULL\b", "", tuple_sql)
    reader = csv.reader(
        StringIO(normalized),
        delimiter=",",
        quotechar="'",
        escapechar="\\",
        skipinitialspace=True,
    )
    return [_coerce(value) for value in next(reader)]


def _coerce(value: str) -> Any:
    if value == "":
        return None
    value = value.replace("\\'", "'")
    try:
        if re.fullmatch(r"-?\d+", value):
            return int(value)
        if re.fullmatch(r"-?(\d+\.?\d*|\.\d+)([eE]-?\d+)?", value):
            return float(value)
    except ValueError:
        pass
    return value
