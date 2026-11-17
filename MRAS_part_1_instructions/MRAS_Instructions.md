# MRAS\_Instructions

## Version 1.1

# 1\. Purpose

The purpose of the system is to ensure that every map tile is processed according to the same scientific basis, calibration, quality-control procedures, and output specification.

The MRAS-system separates the theoretical assessment methodology from the practical execution of the assessment. This document describes the practical execution by providing instructions. The theoretical assessment methodology is given in the files MRAS\_ScientificBasis and MRAS\_Calibration.



# 2\. Input

Each batch consists of a ZIP archive containing **ten PNG map tiles**.

Requirements:

* PNG format
* One map tile per file
* Unique filenames
* One batch processed at a time

# 3\. Loading the assessment environment

Load the following fixed resources before processing:

* `MRAS\_ScientificBasis.md`
* `MRAS\_Calibration.md`
* `json\_schema.json`

# 4\. Conceptual framework

## 4.1 Map readability

Map readability is defined as the degree to which a human map user can comprehend the information presented in a map.

Map readability is related to map legibility.



## 4.2 Readability dimensions

Map readability is assumed to depend on four principal criteria:

* **Amount of information**: total quantity of information in the map.
* **Spatial distribution**: how map information is arranged spatially.
* **Object complexity**: geometric complexity of individual map objects.
* **Graphical resolution**: perceptual quality of the cartographic representation.

Graphical resolution refers to the cartographic design and not to the pixel resolution of the digital image.



# 5\. Assessment philosophy

Each map tile is assessed independently.

The assessment is based exclusively on visible cartographic content.

No comparison between map tiles is allowed during the assessment.

No information outside the tile may be used.

Each tile is treated as an independent observation.

Assessment environment requirement: Each map tile shall be presented individually to the assessor. Batch montages or compressed overviews may be used only for navigation and file management, never for assigning readability scores.



# 6\. Assessment procedure

For every batch:

1. Open the ZIP archive containing ten PNG map tiles.
2. Process the map tiles in alphabetical filename order.
3. Verify displayed filename - Write filename into working memory - Only then begin assessment.
4. Assess each tile independently.
5. Observe only visible cartographic content.

# 7\. Assess Amount of Information.

The criterion is scored on a continuous 1.0–4.0 scale with 1 decimal precision. Use the following rules in your scoring:
1.0 = High
2.0 = Quite high
3.0 = Quite low
4.0 = Low
Write an argument of your score of around 100 characters.
See MRAS\_ScientificBasis.md and MRAS\_Calibration.md for further instructions.

# 8\. Assess Spatial Distribution.

The criterion is scored on a continuous 1.0–4.0 scale with 1 decimal precision. Use the following rules in your scoring:
1.0 = Uneven
2.0 = Quite uneven
3.0 = Quite even
4.0 = Even
Write an argument of your score of around 100 characters.
See MRAS\_ScientificBasis.md and MRAS\_Calibration.md for further instructions.

# 9\. Assess Object Complexity.

The criterion is scored on a continuous 1.0–4.0 scale with 1 decimal precision. Use the following rules in your scoring:
1.0 = High
2.0 = Quite high
3.0 = Quite low
4.0 = Low
Write an argument of your score of around 100 characters.
See MRAS\_ScientificBasis.md and MRAS\_Calibration.md for further instructions.

# 10\. Assess Graphical Resolution.

The criterion is scored on a continuous 1.0–4.0 scale with 1 decimal precision. Use the following rules in your scoring:
1.0 = Poor
2.0 = Quite poor
3.0 = Quite good
4.0 = Good
Image pixel quality is explicitly ignored.
Write an argument of your score of around 100 characters.
See MRAS\_ScientificBasis.md and MRAS\_Calibration.md for further instructions.

# 11\. Synthesize an Overall Map Readability score.

The synthesis of map readability is scored on a continuous 1.0–4.0 scale with 1 decimal precision. Follow instudctions in
the files MRAS\_ScientificBasis.md and MRAS\_Calibration.md to perform the synthesis. Use the following rules in your scoring:
1.0 = Very difficult to read
2.0 = Difficult to read
3.0 = Easy to read
4.0 = Very easy to read
Write an argument of your score of around 200 characters.

# 12\. Produce one JSON object per tile, matching `json\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_schema.json`.

# 13\. Verify:

&#x20;   \* every filename appears exactly once;
\* the batch contains exactly ten maps;
\* arguments are tile-specific;
\* overall map readability scores are tile-specific;
\* the JSON validates against the schema.



# 14\. Save one JSON file for the batch.

For every tile, the observation summary must only describe visible objects, for example:

* buildings
* roads
* paths
* railways
* contour lines
* lakes
* rivers
* streams
* forests
* marshes
* labels
* point symbols
* power lines

# 15\. Batch processing

* Process maps sequentially.
* Use alphabetical filename order.
* Assess each tile independently.

# 16\. Quality control

Before finalizing a batch, verify:

* Every input filename appears exactly once.
* Number of JSON objects equals number of maps.
* JSON validates against the schema.
* Observations describe only visible content.
* Arguments are tile-specific.
* Overall readability is consistent with the four aspect assessments.
* Calibration remains fixed across batches.
* Verify that each JSON object corresponds to the correct image (not only the correct filename).
* Verify that the observation summary matches the visible map content.
* Then perform the existing MRAS duplicate and consistency checks.

# 17\. Output

Each batch produces:

* One JSON assessment file.
* Optional quality-control report.
* Optional execution log.

# 18\. Validation

After all batches have been completed, compare AI assessments with the human user study using:

* Analysis of systematic differences
* Spearman rank correlation
* Mean Absolute Error (MAE)
No recalibration shall be performed after validation begins.

# 19\. Reproducibility

MRAS Version 1.1 is defined by five fixed components:

* `MRAS\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_Overview.md`
* `MRAS\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_Instructions.md`
* `MRAS\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_ScientificBasis.md`
* `MRAS\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_Calibration.md`
* `json\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_schema.json`
Together these define a reproducible AI-assisted workflow for topographic map readability assessment.



# **Appendix A. Strict Batch-by-Batch Assessment Protocol**



This protocol shall be followed for every batch without exception.



No assessment rule may be modified after Batch 1 has been completed.



Step 1. Load fixed resources



Before assessing a batch, load only:



MRAS\_ScientificBasis.md

MRAS\_Calibration.md

json\_schema.json



No other information may be used.



Step 2. Calibration reminder



Before inspecting the first map of every batch, remind yourself of the frozen calibration:



Amount of information is the strongest criterion.

Spatial distribution is the second strongest.

Object complexity contributes moderately.

Graphical resolution contributes least.

Image pixel resolution shall be ignored.



No recalibration is permitted.



Step 3. Process one map at a time



Maps shall never be compared with each other.



For every tile:



Open only the current map.

Ignore all previous maps.

Ignore all following maps.

Perform the complete assessment.

Store the assessment.

Continue to the next map.



Never score several maps simultaneously.



Step 4. Mandatory observation sequence



For every map the observations shall always be performed in exactly this order.



4.1 Observation summary



Describe only visible map content.



For example:



buildings

roads

railways

paths

contour lines

lakes

rivers

marsh

forests

labels

power lines

point symbols



No interpretation.



No scoring.



4.2 Amount of Information



Inspect only



number of objects

occupied map area

density

empty areas

number of object types

regularity of object patterns



Assign score.



Write argument.



4.3 Spatial Distribution



Inspect only



clustering

local crowding

overlap

spatial balance

proximity



Assign score.



Write argument.



4.4 Object Complexity



Inspect only



sinuosity

branching

angularity

transport complexity

polygon complexity



Assign score.



Write argument.



4.5 Graphical Resolution



Inspect only



colour contrast

symbol separation

line visibility

text visibility



Ignore image pixel quality.



Assign score.



Write argument.



4.6 Overall Readability



Only after the four criteria have been completed.



Synthesize the four assessments using the calibrated interpretation.



Do not compute a mathematical average.



Step 5. Mandatory consistency check



Before continuing to the next map verify:



✓ observation summary only contains visible objects



✓ arguments correspond to the observations



✓ overall score is consistent with the four criteria



✓ no criterion has been forgotten



Step 6. Batch validation



After ten maps:



Verify



✓ exactly ten JSON objects



✓ filenames unique



✓ filenames sorted alphabetically



✓ JSON validates



✓ every map assessed independently



✓ arguments are map specific



✓ observation summaries are map specific



Step 7. Duplicate detection



Before finalizing the batch, perform the following checks.



Score duplication



If



all ten maps receive identical overall scores



or



more than five consecutive maps receive identical scores



the batch shall be re-evaluated.



Argument duplication



If two maps have identical arguments, verify that the maps genuinely contain similar cartographic characteristics.



Otherwise rewrite the arguments.



Observation duplication



If two observation summaries are identical, inspect the maps again.



Observation summaries should normally differ because map content differs.



Step 8. Calibration freeze



Never adjust



score interpretation

criterion importance

anchor examples



after Batch 1.



The same decision model shall be applied to every remaining batch.



# Appendix B. Low-Resolution Assessment Rule



Because the available map tiles are low-resolution, apply the following rule consistently:



Assess only what is visibly distinguishable.

Never infer hidden details.

Never penalize a map because the image resolution is low.

Base all scores on the visible cartographic structure.

Apply exactly the same interpretation throughout the entire project.

