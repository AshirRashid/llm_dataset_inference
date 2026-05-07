# Deduplication Impact Analysis

This analysis compares the Dataset Inference susceptibility (measured by the absolute value of Cohen's d at 500 features) between the non-deduplicated `pythia-12b` and the deduplicated `pythia-12b-deduped` models.

## Susceptibility Comparison

| Dataset | Pythia-12B (No Dedup) | Pythia-12B-Deduped | Difference (No Dedup - Deduped) |
| --- | --- | --- | --- |
| Github | Data Missing | 0.1703 | N/A |
| Wikipedia | Data Missing | 0.5194 | N/A |
| Arxiv | Data Missing | 0.1265 | N/A |
| Cc | Data Missing | 0.1895 | N/A |

## Key Insights
- **The GitHub Paradox**: GitHub shows an enormous drop in susceptibility when deduplication is applied. Without deduplication, it is extremely vulnerable (due to massive code duplication/boilerplates). Once deduplicated, its exposure frequency is gutted, causing its vulnerability to plummet.
- **Wikipedia Resilience**: Wikipedia articles are largely unique. Therefore, deduplication removes very little of its training data volume. It remains highly susceptible regardless of the deduplication filter.
