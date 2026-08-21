# Benchmark results

Source workbook: `data/synthetic.xlsx`  
Total runtime: 27.63s  
Scenarios: 7

## Headline

- Greedy over-books **100 hub-scenario pairs** by **242,120 units**; 0 of its plans are executable.
- MILP penalised objective **2.475113** vs repaired-greedy **2.526086** — a **2.02% improvement**, both fully executable.
- Proven optimal on 7 scenarios, mean solve time **0.011s**.
- Split-shipment relaxation lifts coverage from **0.7678** to **0.9106**.
- An 85% service constraint raises mean on-time probability from **0.8408** to **0.9175**, at a coverage cost of **0.2163**.

## Full results

| scenario             | method        |   objective_per_shipment |   mean_score |   coverage |   shipments_assigned |   hub_violations |   excess_units | executable   |   mean_on_time_probability |   mean_cvar95_days | solver_status   |   solve_seconds |
|:---------------------|:--------------|-------------------------:|-------------:|-----------:|---------------------:|-----------------:|---------------:|:-------------|---------------------------:|-------------------:|:----------------|----------------:|
| baseline             | greedy        |                 2.22354  |     0.1771   |     0.7917 |                  190 |               17 |          45510 | False        |                     0.8462 |             8.5433 | greedy-argmin   |          0      |
| baseline             | greedy_repair |                 2.46194  |     0.220891 |     0.7708 |                  185 |                0 |              0 | True         |                     0.8302 |             9.1697 | repaired        |          0      |
| baseline             | milp          |                 2.4067   |     0.202194 |     0.775  |                  186 |                0 |              0 | True         |                     0.8378 |             8.8123 | OPTIMAL         |          0.0132 |
| baseline             | milp_split2   |                 1.60653  |     0.221198 |     0.8583 |                  206 |                0 |              0 | True         |                     0.8441 |             8.5547 | OPTIMAL         |          0.1241 |
| baseline             | milp_split3   |                 1.14     |     0.245877 |     0.9083 |                  218 |                0 |              0 | True         |                     0.8432 |             8.568  | OPTIMAL         |          7.7071 |
| baseline             | milp_sla85    |                 4.79332  |     0.082523 |     0.525  |                  126 |                0 |              0 | True         |                     0.9185 |             6.4849 | OPTIMAL         |          0.0041 |
| port_congestion      | greedy        |                 2.08869  |     0.162104 |     0.8042 |                  193 |               22 |          56580 | False        |                     0.8688 |             7.8681 | greedy-argmin   |          0      |
| port_congestion      | greedy_repair |                 2.35959  |     0.194124 |     0.7792 |                  187 |                0 |              0 | True         |                     0.8654 |             8.0077 | repaired        |          0      |
| port_congestion      | milp          |                 2.27578  |     0.191472 |     0.7875 |                  189 |                0 |              0 | True         |                     0.8652 |             8.066  | OPTIMAL         |          0.0157 |
| port_congestion      | milp_split2   |                 1.63742  |     0.20966  |     0.8542 |                  205 |                0 |              0 | True         |                     0.8733 |             7.7109 | OPTIMAL         |          0.2016 |
| port_congestion      | milp_split3   |                 0.91386  |     0.221195 |     0.9292 |                  223 |                0 |              0 | True         |                     0.8755 |             7.5537 | OPTIMAL         |          1.789  |
| port_congestion      | milp_sla85    |                 3.38621  |     0.140941 |     0.6708 |                  161 |                0 |              0 | True         |                     0.9272 |             6.1632 | OPTIMAL         |          0.0056 |
| cold_chain           | greedy        |                 2.67435  |     0.23247  |     0.75   |                   36 |                4 |          10890 | False        |                     0.803  |             9.9578 | greedy-argmin   |          0      |
| cold_chain           | greedy_repair |                 2.88268  |     0.239103 |     0.7292 |                   35 |                0 |              0 | True         |                     0.7994 |             9.9909 | repaired        |          0      |
| cold_chain           | milp          |                 2.88268  |     0.239103 |     0.7292 |                   35 |                0 |              0 | True         |                     0.8    |             9.9911 | OPTIMAL         |          0.0032 |
| cold_chain           | milp_split2   |                 1.44719  |     0.225365 |     0.875  |                   42 |                0 |              0 | True         |                     0.8175 |             9.4241 | OPTIMAL         |          0.0075 |
| cold_chain           | milp_split3   |                 1.2557   |     0.238925 |     0.8958 |                   43 |                0 |              0 | True         |                     0.8112 |             9.7521 | OPTIMAL         |          0.0148 |
| cold_chain           | milp_sla85    |                 6.06667  |     0.063158 |     0.3958 |                   19 |                0 |              0 | True         |                     0.8929 |             7.4073 | OPTIMAL         |          0.0027 |
| primary_hub_down     | greedy        |                 2.34022  |     0.169265 |     0.7792 |                  187 |               20 |          45860 | False        |                     0.8575 |             8.3304 | greedy-argmin   |          0      |
| primary_hub_down     | greedy_repair |                 2.61024  |     0.201423 |     0.7542 |                  181 |                0 |              0 | True         |                     0.8495 |             8.6162 | repaired        |          0      |
| primary_hub_down     | milp          |                 2.53052  |     0.203962 |     0.7625 |                  183 |                0 |              0 | True         |                     0.8532 |             8.5529 | OPTIMAL         |          0.0128 |
| primary_hub_down     | milp_split2   |                 1.81647  |     0.228625 |     0.8375 |                  201 |                0 |              0 | True         |                     0.8595 |             8.2139 | OPTIMAL         |          0.0569 |
| primary_hub_down     | milp_split3   |                 1.17126  |     0.235493 |     0.9042 |                  217 |                0 |              0 | True         |                     0.8646 |             7.987  | OPTIMAL         |          0.6132 |
| primary_hub_down     | milp_sla85    |                 4.07204  |     0.120062 |     0.6    |                  144 |                0 |              0 | True         |                     0.9272 |             6.1027 | OPTIMAL         |          0.004  |
| air_capacity_reduced | greedy        |                 2.58813  |     0.172104 |     0.7542 |                  181 |               13 |          18120 | False        |                     0.8632 |             8.0674 | greedy-argmin   |          0      |
| air_capacity_reduced | greedy_repair |                 2.72262  |     0.187796 |     0.7417 |                  178 |                0 |              0 | True         |                     0.8514 |             8.4578 | repaired        |          0      |
| air_capacity_reduced | milp          |                 2.65154  |     0.202055 |     0.75   |                  180 |                0 |              0 | True         |                     0.8531 |             8.4036 | OPTIMAL         |          0.0119 |
| air_capacity_reduced | milp_split2   |                 1.63536  |     0.207247 |     0.8542 |                  205 |                0 |              0 | True         |                     0.8603 |             8.0146 | OPTIMAL         |          0.1501 |
| air_capacity_reduced | milp_split3   |                 1.12076  |     0.224687 |     0.9083 |                  218 |                0 |              0 | True         |                     0.8597 |             8.1121 | OPTIMAL         |          1.1016 |
| air_capacity_reduced | milp_sla85    |                 4.11987  |     0.131247 |     0.5958 |                  143 |                0 |              0 | True         |                     0.9254 |             6.2591 | OPTIMAL         |          0.0038 |
| expedite_priority    | greedy        |                 1.9966   |     0.08534  |     0.8072 |                  134 |                9 |          29870 | False        |                     0.8685 |             7.798  | greedy-argmin   |          0      |
| expedite_priority    | greedy_repair |                 2.19038  |     0.103837 |     0.7892 |                  131 |                0 |              0 | True         |                     0.8632 |             7.9802 | repaired        |          0      |
| expedite_priority    | milp          |                 2.13413  |     0.108066 |     0.7952 |                  132 |                0 |              0 | True         |                     0.8629 |             8.0507 | OPTIMAL         |          0.0078 |
| expedite_priority    | milp_split2   |                 1.439    |     0.131071 |     0.8675 |                  144 |                0 |              0 | True         |                     0.8664 |             7.8421 | OPTIMAL         |          0.0361 |
| expedite_priority    | milp_split3   |                 0.986435 |     0.156239 |     0.9157 |                  152 |                0 |              0 | True         |                     0.8618 |             7.9929 | OPTIMAL         |          0.2343 |
| expedite_priority    | milp_sla85    |                 4.5518   |     0.061531 |     0.5482 |                   91 |                0 |              0 | True         |                     0.9132 |             6.5784 | OPTIMAL         |          0.0039 |
| sustainability       | greedy        |                 2.2647   |     0.229096 |     0.7917 |                  190 |               15 |          35290 | False        |                     0.8141 |             9.6632 | greedy-argmin   |          0      |
| sustainability       | greedy_repair |                 2.45516  |     0.264724 |     0.775  |                  186 |                0 |              0 | True         |                     0.8106 |             9.8955 | repaired        |          0      |
| sustainability       | milp          |                 2.44444  |     0.250895 |     0.775  |                  186 |                0 |              0 | True         |                     0.8131 |             9.7346 | OPTIMAL         |          0.0102 |
| sustainability       | milp_split2   |                 1.64915  |     0.27085  |     0.8583 |                  206 |                0 |              0 | True         |                     0.8177 |             9.458  | OPTIMAL         |          0.1124 |
| sustainability       | milp_split3   |                 1.13697  |     0.287089 |     0.9125 |                  219 |                0 |              0 | True         |                     0.8186 |             9.4091 | OPTIMAL         |          1.3146 |
| sustainability       | milp_sla85    |                 4.80505  |     0.104861 |     0.525  |                  126 |                0 |              0 | True         |                     0.9183 |             6.5326 | OPTIMAL         |          0.004  |
