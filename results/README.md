# Results Directory

Demo outputs are saved here when running `python run.py` without `--output-dir`.

```
results/
├── mutation_scores.csv      # All scored mutants (L×19), ranked
├── mutation_heatmap.png     # Fitness landscape heatmap
├── correlation_plot.png     # ESM-2 vs. DMS scatter (if ProteinGym matched)
├── merged_with_dms.csv      # Merged predictions + DMS ground truth
└── mutation_report.txt      # Text summary with top hits
```
