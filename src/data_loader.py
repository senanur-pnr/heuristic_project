from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import mysql.connector
from dotenv import load_dotenv


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


def get_connection():
    load_dotenv()

    return mysql.connector.connect(
        host=os.getenv("DB_HOST", "localhost"),
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASSWORD", ""),
        database=os.getenv("DB_NAME", "modp_db"),
    )


def fetch_all(cursor, query: str):
    cursor.execute(query)
    return cursor.fetchall()


def load_diet_data(sql_path: str | None = None) -> DietData:
    connection = get_connection()
    cursor = connection.cursor(dictionary=True)

    try:
        food_group_rows = fetch_all(cursor, "SELECT id, name FROM food_group")
        nutrient_rows = fetch_all(cursor, "SELECT id, name FROM nutrients")
        food_nutrient_rows = fetch_all(cursor, "SELECT foodId, nutrientId, quantity FROM food_nutrients")
        food_rows = fetch_all(cursor, "SELECT * FROM foods")
        user_rows = fetch_all(cursor, "SELECT * FROM user")
        user_food_rows = fetch_all(cursor, "SELECT * FROM user_foods")
        dri_db_rows = fetch_all(cursor, "SELECT * FROM dri")

        food_groups = {int(row["id"]): str(row["name"]) for row in food_group_rows}
        nutrients = {int(row["id"]): str(row["name"]) for row in nutrient_rows}

        food_nutrients: dict[int, dict[int, float]] = {}
        for row in food_nutrient_rows:
            food_id = int(row["foodId"])
            nutrient_id = int(row["nutrientId"])
            quantity = float(row["quantity"] or 0)
            food_nutrients.setdefault(food_id, {})[nutrient_id] = quantity

        foods: dict[int, Food] = {}
        for row in food_rows:
            food_id = int(row["id"])
            foods[food_id] = Food(
                id=food_id,
                name=str(row["name"]),
                food_group_id=int(row["foodGroupId"]),
                portion=int(row["portion"] or 0),
                cost=float(row["cost"] or 0),
                preference=float(row["preference"] or 0),
                preference2=float(row["preference2"] or 0),
                preparing_time=float(row["preparingTime"] or 0),
                cooking_time=float(row["cookingTime"] or 0),
                rating=float(row["rating"] or 0),
                co2=float(row["co2"] or 0),
                availability=int(row["availability"] or 0),
                nutrients=food_nutrients.get(food_id, {}),
            )

        users = [
            User(
                id=int(row["id"]),
                name=str(row["name"] or ""),
                surname=str(row["surname"] or ""),
                username=str(row["username"] or ""),
                age=int(row["age"] or 30),
                gender=str(row["gender"] or "adult"),
                height=int(row["height"] or 170),
                weight=int(row["weight"] or 70),
            )
            for row in user_rows
        ]

        user_preferences: dict[int, dict[int, float]] = {}
        for row in user_food_rows:
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
            for row in dri_db_rows
        ]

        return DietData(
            foods=foods,
            food_groups=food_groups,
            nutrients=nutrients,
            dri_rows=dri_rows,
            users=users,
            user_preferences=user_preferences,
        )

    finally:
        cursor.close()
        connection.close()