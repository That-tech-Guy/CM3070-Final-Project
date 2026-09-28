# Two-stage recommender results

## Controlled findings

- Best Stage 1 topic method: **Prerequisite-ready weakness** (normalized regret 0.349).
- Best Stage 2 resource method: **Difficulty match** (normalized regret 0.103).
- Best deployable end-to-end pipeline: **Rule pipeline** (normalized regret 0.315).
- Statistical comparison: RF two-stage vs Rule pipeline; mean difference 0.0595, p=1.098e-06.
- Across the 8 alternative rule settings, topic agreement with the default ranges from 73.9% to 100.0%, and full-choice agreement ranges from 51.1% to 80.5%.
- A blinded adult-educator review form was generated with 10 synthetic cases; its method key is stored separately.

## Interpretation boundary

The redesign makes topic and resource decisions separately and allows graph features to vary across topic candidates. It still evaluates algorithms against a disclosed simulator and cannot establish improved learning for real students.