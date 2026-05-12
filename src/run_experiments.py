from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from algorithms import RunResult, run_nsga2, run_spea2
from data_loader import load_diet_data
from modp import DietProblem, NUTRIENT_IDS, User, hypervolume_2d_or_3d


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MODP diet optimization experiments.")
    parser.add_argument("--sql", required=True, help="Path to the provided diet MySQL dump.")
    parser.add_argument("--out", default="results", help="Output directory for CSV/JSON/SVG results.")
    parser.add_argument("--generations", type=int, default=35)
    parser.add_argument("--population", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--objectives",
        default="preference,cost,time",
        help="Comma-separated objectives. Preference is mandatory; choose cost,time,co2 as the others.",
    )
    args = parser.parse_args()

    objectives = tuple(part.strip() for part in args.objectives.split(",") if part.strip())
    if "preference" not in objectives or len(objectives) < 3:
        raise SystemExit("Use at least three objectives and include preference, e.g. preference,cost,time")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_diet_data(args.sql)
    users = first_two_users_with_roles(data.users)

    all_rows: list[dict[str, object]] = []
    hv_rows: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []
    run_records: list[dict[str, object]] = []

    for user in users:
        for diversity in (False, True):
            problem = DietProblem(
                data,
                user,
                objectives,
                diversity=diversity,
                vegetarian_override=(user.id == users[1].id),
            )
            runs = [
                run_nsga2(problem, args.generations, args.population, args.seed),
                run_spea2(problem, args.generations, args.population, args.seed + 1000),
            ]
            for run in runs:
                tag = f"user{user.id}_{run.algorithm.lower().replace('-', '')}_{'diversity' if diversity else 'nodiversity'}"
                run_records.append(
                    {
                        "problem": problem,
                        "user": user,
                        "run": run,
                        "tag": tag,
                        "diversity": diversity,
                    }
                )

    reference_point = common_reference_point([record["run"] for record in run_records])

    for record in run_records:
        problem = record["problem"]
        user = record["user"]
        run = record["run"]
        tag = str(record["tag"])
        diversity = bool(record["diversity"])
        assert isinstance(problem, DietProblem)
        assert isinstance(user, User)
        assert isinstance(run, RunResult)
        run.hypervolume_history = [
            hypervolume_2d_or_3d(front, reference_point)
            for front in run.front_history
        ]
        all_rows.extend(menu_rows(problem, run, tag, diversity))
        hv_rows.extend(hypervolume_rows(user, run, tag, diversity, reference_point))
        write_sample_menu_table(problem, run, out_dir / f"{tag}_sample_menus.csv")
        write_front_plot(run, objectives, out_dir / f"{tag}_pareto.svg", tag)
        summary.append(
            {
                "user_id": user.id,
                "username": user.username,
                "detected_vegetarian": problem.is_vegetarian,
                "algorithm": run.algorithm,
                "diversity": diversity,
                "pareto_count": len(run.pareto),
                "last_hypervolume": run.hypervolume_history[-1] if run.hypervolume_history else 0,
                "common_reference_point": reference_point,
                "breakfast_pool": len(problem.breakfast_pool),
                "lunch_dinner_pool": len(problem.lunch_dinner_pool),
            }
        )

    write_csv(out_dir / "pareto_solutions.csv", all_rows)
    write_csv(out_dir / "hypervolume_history.csv", hv_rows)
    write_json(out_dir / "summary.json", summary)
    write_hv_plot(hv_rows, out_dir / "hypervolume_comparison.svg")
    print(f"Done. Results written to {out_dir.resolve()}")


def first_two_users_with_roles(users: list[User]) -> list[User]:
    if len(users) < 2:
        raise ValueError("The users table must contain at least two users.")
    return users[:2]


def common_reference_point(runs: list[object]) -> tuple[float, ...]:
    all_points: list[tuple[float, ...]] = []
    for run in runs:
        assert isinstance(run, RunResult)
        for front in run.front_history:
            all_points.extend(front)
        all_points.extend(ind.menu.fitness for ind in run.pareto)
    if not all_points:
        return (1.0, 1.0, 1.0)
    dimensions = len(all_points[0])
    worst = [max(point[i] for point in all_points) for i in range(dimensions)]
    return tuple(value + max(abs(value) * 0.10, 1e-9) for value in worst)


def menu_rows(problem: DietProblem, run: RunResult, experiment: str, diversity: bool) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for rank, individual in enumerate(run.pareto, start=1):
        menu = individual.menu
        row: dict[str, object] = {
            "experiment": experiment,
            "algorithm": run.algorithm,
            "user_id": problem.user.id,
            "username": problem.user.username,
            "vegetarian": problem.is_vegetarian,
            "diversity_enabled": diversity,
            "rank": rank,
            "food_count": len(menu.all_food_ids),
            "breakfast_ids": " ".join(map(str, menu.breakfast_ids)),
            "breakfast_names": " | ".join(problem.food_names(menu.breakfast_ids)),
            "lunch_dinner_ids": " ".join(map(str, menu.lunch_dinner_ids)),
            "lunch_dinner_names": " | ".join(problem.food_names(menu.lunch_dinner_ids)),
            "distinct_groups": menu.distinct_groups,
            "violated_nutrients": menu.violations,
            "penalty": round(menu.penalty, 6),
            "forbidden_items": menu.vegetarian_forbidden_count,
        }
        for name, value in menu.objectives.items():
            row[f"objective_{name}"] = round(value, 4)
        for label, nid in NUTRIENT_IDS.items():
            low, high = problem.dri_bounds(nid)
            value = menu.nutrients[nid]
            row[f"{label}_value"] = round(value, 4)
            row[f"{label}_RLL"] = low
            row[f"{label}_RUL"] = high
            row[f"{label}_ok"] = low <= value <= high
        rows.append(row)
    return rows


def hypervolume_rows(
    user: User,
    run: RunResult,
    experiment: str,
    diversity: bool,
    reference_point: tuple[float, ...],
) -> list[dict[str, object]]:
    return [
        {
            "experiment": experiment,
            "algorithm": run.algorithm,
            "user_id": user.id,
            "diversity_enabled": diversity,
            "generation": generation,
            "hypervolume": value,
            "reference_point": json.dumps(reference_point),
        }
        for generation, value in enumerate(run.hypervolume_history, start=1)
    ]


def write_sample_menu_table(problem: DietProblem, run: RunResult, path: Path) -> None:
    rows = menu_rows(
        problem,
        RunResult(run.algorithm, run.population, run.pareto[:3], run.hypervolume_history, run.front_history),
        path.stem,
        problem.diversity,
    )
    write_csv(path, rows)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def write_front_plot(run: RunResult, objectives: tuple[str, ...], path: Path, title: str) -> None:
    points = [
        (ind.menu.objectives[objectives[1]], ind.menu.objectives[objectives[2]], ind.menu.objectives["preference"])
        for ind in run.pareto
    ]
    svg = scatter_svg(points, f"{title}: {objectives[1]} vs {objectives[2]}", objectives[1], objectives[2])
    path.write_text(svg, encoding="utf-8")


def write_hv_plot(rows: list[dict[str, object]], path: Path) -> None:
    series: dict[str, list[tuple[int, float]]] = {}
    for row in rows:
        key = str(row["experiment"])
        series.setdefault(key, []).append((int(row["generation"]), float(row["hypervolume"])))
    path.write_text(line_svg(series, "Hypervolume by generation", "generation", "hypervolume"), encoding="utf-8")


def scatter_svg(points: list[tuple[float, float, float]], title: str, x_label: str, y_label: str) -> str:
    width, height, pad = 720, 480, 60
    if not points:
        points = [(0, 0, 0)]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    prefs = [p[2] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_p, max_p = min(prefs), max(prefs)

    def sx(x: float) -> float:
        return pad + (x - min_x) / (max_x - min_x or 1) * (width - 2 * pad)

    def sy(y: float) -> float:
        return height - pad - (y - min_y) / (max_y - min_y or 1) * (height - 2 * pad)

    circles = []
    for x, y, pref in points:
        shade = int(220 - (pref - min_p) / (max_p - min_p or 1) * 150)
        circles.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="5" fill="rgb(30,{shade},150)" opacity="0.8"/>')
    return svg_shell(width, height, title, x_label, y_label, "\n".join(circles))


def line_svg(series: dict[str, list[tuple[int, float]]], title: str, x_label: str, y_label: str) -> str:
    width, height, pad = 820, 500, 60
    all_points = [point for points in series.values() for point in points] or [(0, 0)]
    min_x, max_x = min(p[0] for p in all_points), max(p[0] for p in all_points)
    min_y, max_y = min(p[1] for p in all_points), max(p[1] for p in all_points)
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b", "#7f7f7f"]

    def sx(x: float) -> float:
        return pad + (x - min_x) / (max_x - min_x or 1) * (width - 2 * pad)

    def sy(y: float) -> float:
        return height - pad - (y - min_y) / (max_y - min_y or 1) * (height - 2 * pad)

    body = []
    for i, (name, points) in enumerate(sorted(series.items())):
        color = colors[i % len(colors)]
        path = " ".join(f"{'M' if j == 0 else 'L'} {sx(x):.1f} {sy(y):.1f}" for j, (x, y) in enumerate(points))
        body.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2"/>')
        body.append(f'<text x="{width - pad + 8}" y="{pad + 16 * i}" font-size="10" fill="{color}">{name}</text>')
    return svg_shell(width, height, title, x_label, y_label, "\n".join(body))


def svg_shell(width: int, height: int, title: str, x_label: str, y_label: str, body: str) -> str:
    pad = 60
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
  <text x="{width / 2}" y="28" text-anchor="middle" font-family="Arial" font-size="18">{title}</text>
  <line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" stroke="#222"/>
  <line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height - pad}" stroke="#222"/>
  <text x="{width / 2}" y="{height - 16}" text-anchor="middle" font-family="Arial" font-size="13">{x_label}</text>
  <text transform="translate(18 {height / 2}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="13">{y_label}</text>
  {body}
</svg>
'''


if __name__ == "__main__":
    main()
