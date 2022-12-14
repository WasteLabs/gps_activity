# **Architecture** βοΈ

Given part of documentation explains sequence of generic algorithm operations. Generic means that step components can be replaced by different instances and reused.

## **Steps** π£

----

1. Preprocessing gps & plan π 
2. GPS cluster aggregation pipelineπ
3. Spatial joiner π
4. Spatial validator π΅ π£
5. coverage stats extractor π

## 1. **Preprocessing** π 

1. Projects schemas

**Goal**: To deliver required inputs for next modules

## 2. **Cluster aggregation** π

Aggregates cluster centroids

## 3. **Spatial Joiner** π

Performs spatial join of aggregated clusters & preprocessed plan based on maximum spatial distance

## 4. **Spatial validator** π΅ π£

Applies additional constrains to joined plan & clusters:

1. (Optional) overlap of between plan & cluster over `source_vehicle_id`
2. Validates of joins are in the range of maximum allowed days distance

## 5. **Coverage statistics** π

Estimates linkage over vehicle-date between provided gps & plans
*NOTE*: This is needed in metrics module for validation 1-to-1 match
Otherwise, `recall` and `precision` can be biased
