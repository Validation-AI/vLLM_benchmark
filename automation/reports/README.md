Versioned human-readable serving_tuning reports are stored here.

Expected layout:

- `serving_tuning/serving_tuning_results_<date>_build<build>.csv`
- `serving_tuning/serving_tuning_results_<date>_build<build>.xlsx`

The legacy compatibility copies in `logs/` are still produced, but these
versioned report files are the durable report artifacts for each run.
