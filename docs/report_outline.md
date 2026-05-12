# Report Outline

1. Problem formulation
   - Daily menu as a Multi-Objective Multidimensional Knapsack Problem.
   - Objectives: maximize preference, minimize cost, minimize time.
   - Constraints: Energy, Protein, Carbohydrate, Fiber, Sodium.

2. Data loading
   - Read all values from the SQL dump.
   - Use `user_foods.preference` when available.
   - Use `foods.preference` / `foods.preference2` fallback when user preferences are absent.

3. Chromosome
   - Two independent permutations.
   - Breakfast groups in part 1.
   - Lunch+dinner groups in part 2.

4. Decoder
   - Greedy left-to-right.
   - Breakfast checks Energy and Protein at 35% DRI.
   - Lunch+dinner checks all five nutrients.
   - Uses 0.90 lower and 1.15 upper tolerance during decoding.

5. Penalty
   - Under-bound violation: weight 0.7.
   - Over-bound violation: weight 0.3.
   - Diversity penalty when enabled.
   - Huge penalty/skip for `-1` preferences.

6. Algorithms
   - NSGA-II.
   - SPEA2.
   - Common OX crossover and swap mutation.

7. Experiments
   - User comparison.
   - Algorithm comparison.
   - Diversity on/off comparison.

8. Results
   - Pareto fronts.
   - Hypervolume convergence.
   - Three sample menus with nutrient totals.

9. Discussion
   - Trade-offs between preference, cost, and time.
   - Vegetarian vs non-vegetarian menu differences.
   - Effect of diversity penalty.
