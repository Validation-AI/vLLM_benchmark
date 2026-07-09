Automation inputs and generated artifacts for benchmark job orchestration.

Layout:

- `manifests/`
  Human-edited source inputs. Put Excel files here.
- `history/`
  Durable machine-written tuning history. Keep latest state and immutable run snapshots here.
- `reports/`
  Versioned human-readable CSV/XLSX report copies for each build.
- `generated/`
  Machine-generated JSON manifests derived from source inputs.
- `schema.md`
  Field definitions for spreadsheet columns and parsed output.

Current convention:

- Workbook path is fixed at `automation/manifests/serving_tuning/automation_v0.xlsx`.
- The `serving_tuning` sheet is the source of truth for row-specific model/topology fields.
- Jenkins keeps job-level defaults such as `length_configs`, `extra_args`, `docker_name`, `vllm_branch`, `dtype`, and `hardware`.
- The automation flow parses Excel into normalized JSON first, then lets Groovy iterate rows.
- Compatibility outputs are still written to `logs/summary.log`, `logs/row_results/`, `logs/serving_tuning_results.csv`, and `logs/serving_tuning_results.xlsx`.
- Durable history now lives under `automation/history/serving_tuning/`:
  - `latest/row_state.json` for the latest per-row state used by automation
  - `runs/full/` for immutable full per-build row snapshots
  - `runs/best/` for immutable best-per-row summaries used for writeback
- Versioned report copies now live under `automation/reports/serving_tuning/`.
- Manifest-mode tuning now uses persisted `last_batch_size` as a fallback warm-start seed; exact-match priors from `tp_batchsize.json` / `ld_batchsize.json` still take precedence.
