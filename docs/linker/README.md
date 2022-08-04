# **Architecture** ⚙️

Given part of documentation explains sequence of generic algorithm operations. Generic means that step components can be replaced by different instances and reused.

## **Steps** 👣

----

1. Preprocessing gps & plan 🛠
2. GPS cluster aggregation pipeline📍
3. Spatial joiner 🖇
4. Spatial validator 🔵 🟣
5. coverage stats extractor 📊

## 1. **Preprocessing** 🛠

1. Projects schemas

**Goal**: To deliver required inputs for next modules

## 2. **Cluster aggregation** 📍

Aggregates cluster centroids

## 3. **Spatial Joiner** 🖇

Performs spatial join of aggregated clusters & preprocessed plan based on maximum spatial distance

## 4. **Spatial validator** 🔵 🟣

Applies additional constrains to joined plan & clusters:

1. (Optional) overlap of between plan & cluster over `source_vehicle_id`
2. Validates of joins are in the range of maximum allowed days distance

## 5. **Coverage statistics** 📊

Estimates linkage over vehicle-date between provided gps & plans
*NOTE*: This is needed in metrics module for validation 1-to-1 match
Otherwise, `recall` and `precision` can be biased
