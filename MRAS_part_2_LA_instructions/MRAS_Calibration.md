# Calibration Document for AI-Based Assessment of Topographic Map Readability

## Version 1.0

## 1\. Purpose

This document describes the calibration procedure used for the
AI-assisted assessment of topographic map readability.

The objective of the calibration is to ensure that the AI-generated
readability assessments are consistent with expert human judgements
while remaining grounded in established cartographic theory.

The calibration is performed **once** before the assessment of the
complete dataset.

After calibration, the decision model is frozen and applied unchanged to
all map tiles.

\---

# 2\. Calibration dataset

Calibration was performed using sixteen topographic map tiles that were
independently assessed in a human user study.

The calibration maps are not included in the final evaluation dataset.

Filename              Human map readability

Map1.png			3
Map2.png			3.5
Map3.png			3
Map4.png			2
Map5.png			2
Map6.png			3
Map7.png			2
Map8.png			2
Map9.png			3
Map10.png			3
Map11.png			3.5
Map12.png			3.5
Map13.png			2
Map14.png			2.5
Map15.png			3.5
Map16.png			4

\---

# 3\. Calibration procedure

1. Independently inspect every calibration map.
2. Compare the AI assessment with the human readability score.
3. Identify systematic differences.
4. Modify the interpretation of the readability dimensions until the
assessments become consistent with the human study.

The calibration focused on the interpretation of the assessment criteria
rather than numerical score adjustment.

\---

# 4\. Calibration results

## 4.1 Amount of information

Amount of information is the dominant determinant of readability. Large
numbers of visible objects generally reduce readability regardless of
graphical quality.

## 4.2 Spatial distribution

Local concentrations of information reduce readability more than evenly
distributed information.

## 4.3 Object complexity

Transport infrastructure contributes strongly to perceived complexity.
Motorway interchanges, railway yards, multiple parallel roads and
numerous intersections are penalized more strongly than regularly
arranged buildings.

## 4.4 Graphical resolution

Graphical resolution has comparatively little influence on the overall
readability because all maps originate from Lantmäteriet and exhibit
consistently high cartographic quality.

\---

# 5\. Calibrated synthesis

Dimension                 Relative importance

\---

Amount of information                     45%
Spatial distribution                      25%
Object complexity                         20%
Graphical resolution                      10%

These values are conceptual weights rather than mathematical
coefficients.

\---

# 6\. Anchor examples

## High readability (approximately 3.3--3.6)

Typical characteristics:

* sparse information content
* little clustering
* simple object geometry
* very limited visual clutter

## Medium readability (approximately 2.4--3.0)

Typical characteristics:

* moderate information density
* some local clustering
* several object classes
* moderate transport infrastructure

## Low readability (approximately 1.6--2.2)

Typical characteristics:

* very high information density
* extensive transport infrastructure
* local congestion
* multiple parallel linear features
* high cognitive load

\---

# 7\. Decision rules

1. Assess each tile independently.
2. Use only visible cartographic content.
3. Ignore image pixel quality.
4. Evaluate graphical resolution only as cartographic design.
5. Give greatest importance to amount of information.
6. Penalize local clustering.
7. Penalize transport complexity more than regular urban grids.
8. Do not compare maps within or between batches.
9. Keep the calibrated interpretation fixed throughout the project.

\---

# 8\. Calibration freeze

The calibration is performed only once.

After calibration has been completed, no further adjustment of the
decision model is permitted during the assessment of the complete
dataset.

This prevents calibration drift and ensures that every map is assessed
according to exactly the same interpretation.

