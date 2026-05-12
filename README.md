# Multi-Objective Diet Optimization Project

This repository implements the MODP term project over the provided `diet (1).sql` dump.

## What Is Implemented

- Database-backed loader for `foods`, `food_group`, `food_nutrients`, `nutrients`, `dri`, `user`, and `user_foods`.
- Two-part chromosome:
  - Part 1: breakfast food groups.
  - Part 2: lunch+dinner food groups.
- Custom greedy decoder:
  - Breakfast uses Energy and Protein with 35% DRI split.
  - Lunch+dinner continues from breakfast totals and checks all 5 required nutrients.
  - Upper tolerance is 1.15 and lower tolerance is 0.90.
- Custom penalty function:
  - 0.7 under-nutrition penalty.
  - 0.3 over-nutrition penalty.
  - Optional diversity penalty.
  - Huge penalty/skip for `-1` preferences and vegetarian-forbidden meat items.
- Two algorithms implemented from scratch:
  - NSGA-II
  - SPEA2
- Required experiment modes:
  - User 1 vs User 2.
  - NSGA-II vs SPEA2.
  - Diversity disabled vs enabled.
- CSV/JSON/SVG outputs.

## Run

Use Python 3.10+.

```bash
python3 src/run_experiments.py \
  --sql "/Users/senapinarbasi/Downloads/diet (1).sql" \
  --out results \
  --generations 35 \
  --population 50
```

The default objectives are:

```text
preference,cost,time
```

You can switch the second minimization objective:

```bash
python3 src/run_experiments.py --sql "/path/to/diet.sql" --objectives preference,cost,co2
```

## Outputs

- `results/pareto_solutions.csv`: all Pareto menu solutions.
- `results/hypervolume_history.csv`: convergence data per generation.
- `results/summary.json`: run summary, detected user roles, pool sizes.
- `results/*_sample_menus.csv`: at least 3 sample menus per experiment.
- `results/*_pareto.svg`: Pareto scatter plots.
- `results/hypervolume_comparison.svg`: convergence comparison.

## Five-Person Work Split

1. SQL/Data owner:
   - Verify `foods`, `food_group`, `food_nutrients`, `dri`, `user`, `user_foods`.
   - Document which food groups map to breakfast and lunch+dinner.
   - Confirm User 1/User 2 preference availability and vegetarian detection.

2. Chromosome/Decoder owner:
   - Explain the two-part permutation representation.
   - Explain breakfast/lunch+dinner greedy decoding.
   - Validate nutrient totals against DRI bounds.

3. Algorithm owner 1:
   - Explain and test NSGA-II.
   - Cover non-dominated sorting, crowding distance, binary tournament, OX crossover, swap mutation.

4. Algorithm owner 2:
   - Explain and test SPEA2.
   - Cover strength fitness, density, archive selection, truncation.

5. Experiment/Report owner:
   - Run all experiments.
   - Prepare Pareto plots, hypervolume plot, and sample menu tables.
   - Write discussion and compare diversity on/off.

## Notes For The Report

The project handout allows using libraries for MOEA, but this code implements NSGA-II and SPEA2 directly. The chromosome representation, decoding, and penalty are custom code as required.

The first two rows in the `user` table are used as User 1 and User 2. Following the handout, User 1 is run as non-vegetarian and User 2 is run as vegetarian. If explicit preferences are missing in another dataset, the loader can fall back to the `foods.preference` / `foods.preference2` columns, with vegetarian detection based on `-1` ratings for meat-based foods.
