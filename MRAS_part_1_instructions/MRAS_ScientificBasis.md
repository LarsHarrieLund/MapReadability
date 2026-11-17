# Scientific Basis for MRAS Version 2.0

## Purpose

This document provides the scientific foundation of the Map Readability
Assessment System (MRAS). Unlike the calibration document, which is
dataset-specific, this document synthesizes the cartographic,
perceptual, and map generalisation literature into a reusable
theoretical framework for assessing map readability across different map
types.

The document is organised around a conceptual progression:

**Map readability theory → Four readability criteria → Analytical
measures → Composite measures → Expert synthesis → MRAS**



# 1\. Map Readability Theory

MRAS study map readability from a bottom-up perspective, focusing on
the graphical characteristics of the map itself. Such approaches assume
that readability is primarily determined by visual properties of the map,
including symbol density, graphical complexity, and visual organization.
However, map use also involves top-down cognitive processes that are
task-dependent and draw upon prior knowledge stored in long-term memory
and understand the semantic meaning of the symbols in the map. Consequently,
a comprehensive understanding of map use requires consideration of both
perceptual and cognitive factors. Nevertheless, in MRAS the concept of
map readability is restricted to the bottom-up perspective.



# 2\. Four Readability Criteria

## 2.1 Amount of Information

The literature consistently identifies amount of information as the
strongest determinant of readability. Common analytical measures include
number of objects, number of vertices, object line length, occupied
space and number of object types.

Studies of map readability have demonstrated that measures of
amount of information had the highest explanatory power for perceived
readability, with object line length, number of object types and number
of vertices performing particularly well.

Information that are in regular patterns are easier to read than irregular patterns.

Therefore, a dense regular network is easy to read even though it contains much

information.

The assessment of amount of information should consider
•	number of visible objects
•	density of objects
•	occupied map area
•	presence of empty space

•	if the object patterns are regular.



## 2.2 Spatial Distribution

Readability depends not only on how much information is present but also
on how it is organised. Local clustering, crowding, overlap and
proximity increase perceptual difficulty even if the total information
amount is unchanged.

The strongest analytical measures include proximity between disjoint objects,
local density, degree of overlap and entropy-based spatial
distribution.

The assessment of spatial distribution should consider
•	clustering
•	local concentrations
•	spatial balance
•	crowding
•	overlap.

## 2.3 Object Complexity

Object complexity reflects the geometric intricacy of individual map
objects. The literature indicates that these measures contribute to readability,
although their influence is generally weaker than amount of information
and spatial distribution for already-generalised topographic maps.
For automatically produced maps the influence is more uncertain.

The assessment of object complexity should consider
•	sinuosity
•	angularity
•	branching
•	geometric detail
•	polygon boundary complexity.

## 2.4 Graphical Resolution

Graphical resolution concerns symbol discriminability rather than image
resolution.

For standardised topographic maps graphical resolution often contributes
less to variability than the first two criteria, although poor colour
contrast or insufficient separation can still reduce readability.

The assessment of graphical resolution should consider
•	colour contrast
•	brightness
•	line width
•	symbol visibility
•	text visibility.

## 2.5 Note

Visual clutter is not treated as an independent criterion in MRAS.
Instead, clutter is interpreted as the performance consequence of
amount of information and spatial distribution.



# 3\. Analytical Measures

The literature proposes numerous analytical measures, but they can be
grouped naturally under the four criteria.

Amount of information: - number of objects – number of object types –
number of vertices – line length - occupied area

Spatial distribution: - Voronoi entropy - local density - proximity -
overlap - neighbourhood relations

Object complexity: - sinuosity - angularity - connectivity - object
size - segment length

Graphical resolution: - colour contrast - brightness difference - hue
difference - minimum separations

The Harrie et al. 2015 study provides a comprehensive comparison of these
measures against human judgement and therefore forms the methodological backbone
of MRAS.



# 4\. Composite Measures

A central conclusion of the literature is that **no single analytical
measure adequately describes map readability**.

The Harrie et al. 2015 demonstrated that composites consistently outperform
individual measures. Approximately 80% of map samples could be classified correctly
using combinations of object line length, number of object types and
proximity indicator.

This result is consistent with Moacdieh \& Sarter, who recommend
combining complementary measurement approaches because clutter and
readability are multidimensional constructs.



# 5\. Expert Synthesis

Although analytical composites improve prediction, they do not
completely explain readability.

The literature identifies several reasons:

* semantic understanding influences perception,
* familiar geographic patterns are easier to read,
* unusual contexts increase cognitive effort,
* readability depends partly on task and experience.

Therefore MRAS adopts **expert synthesis** rather than a purely
mathematical formula.

Expert synthesis integrates: - the analytical measures, - cartographic
expertise, - perceptual interpretation, - contextual judgement.

\---

# 6\. MRAS

MRAS operationalises the scientific literature through four fixed
readability criteria:

1. Amount of Information
2. Spatial Distribution
3. Object Complexity
4. Graphical Resolution

Each criterion is evaluated independently.

The final readability assessment is **not** computed by a fixed weighted
equation.

Instead, the four criteria are synthesised by expert reasoning informed
by the analytical evidence and the scientific literature.

This architecture follows the conclusions of the readability literature:

* readability is multidimensional;
* analytical measures are necessary;
* composite measures outperform single measures;
* expert synthesis provides the most robust overall assessment.

Consequently, MRAS should be regarded as a scientifically grounded
framework for human-centred bottom-up map readability assessment
rather than as a single numerical readability index.

