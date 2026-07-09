## Serving Tuning Sheet

Worksheet name:

- `serving_tuning`

Row-driving columns:

- `row_id`
  Stable identifier for reruns, logs, and compiled outputs.
- `enabled`
  `TRUE` runs the row. `FALSE` skips it without deleting the row.
- `model_id`
  Hugging Face model id.
- `disk_id`
  Logical cache disk selector, for example `2` -> `/localdisk2`.
- `tp`
  Tensor parallel size.
- `pp`
  Pipeline parallel size.
- `dp`
  Data parallel size.
- `dp_mode`
  Current project rule: if `dp > 1`, use `router_dp`; otherwise `none`.
- `notes`
  Human comments only.

Legacy compatibility columns still tolerated by the parser:

- `hugginface_path`
  If blank, derive from `disk_id`.
- `parallel_spec`
  Ignored when explicit `tp/pp/dp` are present.
- `source_sheet`
- `source_row`
- prior metric columns such as `prior_batch_size`, `prior_ttft_ms`, `prior_tpot_ms`, `prior_throughput_tok_s`

## Jenkins-Owned Defaults

These are not row-specific and should stay in Jenkins for this workflow:

- `benchmark_script=serving_tuning`
- `length_configs`
- `extra_args`
- `docker_name`
- `vllm_branch`
- `dtype`
- `hardware`
- `tuning_slas`

`extra_args` is still read from Jenkins, but topology flags are normalized from workbook rows.

## Parsing Rules

### `enabled`

- `TRUE` / `true` / `1` means run the row
- anything else means skip unless explicitly included by tooling

### `disk_id`

Map logical disk ids to cache paths before launch:

- `1` => `/localdisk1`
- `2` => `/localdisk2`
- `3` => `/localdisk3`
- `4` => `/localdisk4`

### `tp` / `pp` / `dp`

These are the source of truth for topology in the new design.

If legacy `parallel_spec` is still present:

- it is only used as a fallback when one of `tp/pp/dp` is missing

### `dp_mode`

Current convention:

- `dp == 1` => `dp_mode=none`
- `dp > 1` => `dp_mode=router_dp`

## Outputs

The automated flow writes:

- inspectable Jenkins log lines in `logs/summary.log`
- per-row JSON result files in `logs/row_results/`
- merged CSV report at `logs/serving_tuning_results.csv`
- merged workbook report at `logs/serving_tuning_results.xlsx`
- latest per-row state at `automation/history/serving_tuning/latest/row_state.json`
- immutable full run history at `automation/history/serving_tuning/runs/full/`
- immutable best-per-row snapshots at `automation/history/serving_tuning/runs/best/`
- versioned report copies at `automation/reports/serving_tuning/`

Current behavior note:

- prior history is stored and readable by automation
- manifest-mode tuning uses persisted `last_batch_size` as a fallback seed when no exact fingerprint match exists in `tp_batchsize.json` / `ld_batchsize.json`

Recommended compiled result fields:

- `row_id`
- `model_id`
- `disk_id`
- `tp`
- `pp`
- `dp`
- `dp_mode`
- `dtype`
- `hardware`
- `length_config`
- `status`
- `best_batch_size`
- `throughput`
- `ttft_ms`
- `tpot_ms`
- `error`
- `build_number`
- `build_url`
- `server_log`
- `client_log`
