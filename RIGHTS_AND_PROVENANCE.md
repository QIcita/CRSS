# Rights and provenance

This is a private research snapshot for code inspection and manuscript collaboration. No new blanket open-source license is granted by this export. Existing copyright and license notices in bundled source and the CVPR template remain applicable. Dataset access and third-party dependencies follow their original terms.

- `code/`: byte-preserving copy of the 2026-10-11 full-model release; all 582 manifest entries are retained, including original checkpoints and task-separated source snapshots.
- `paper/`: tracked files from the Overleaf commit in `provenance/source_map.json`, plus its compiled PDF. Overleaf Git history and credentials are not exported.
- `docs/source_reports/` and `evidence/`: selected dated research reports, configuration/result audits and citation records; these are evidence snapshots, not newly executed experiments.
- `docs/` reading guides and root README: repository navigation added for this export; they do not change model computations or overwrite frozen reports.

Raw datasets, full geometry caches, credentials, account configuration, SSH keys and third-party paper PDFs are not included. Absolute server paths inside frozen code and evidence are preserved for provenance, not as a promise of portable training.

## R9 update (2026-10-17)

`code/r9/` adds the frozen R9 source, two retained H01 checkpoints, all six H01 result records, and task-extension code/protocols/acceptance/status snapshots. See `code/r9/server_export.json` for byte-level source mapping and explicit publication omissions. Inactive upstream demo datasets/arrays/figures and generic Python runtime packages are omitted; original protocol files and their historical hashes are preserved. Utonia, GeoTransformer and Polynormer copyright/license notices are retained. Only Utonia executes in the selected R9 configuration. No blanket license change is made. No live optimizer checkpoints, raw neuronal datasets or geometry caches are included.

`docs/R9_*` and root navigation describe the new version. Historical model release bytes and paper sources are preserved; the paper has not been rewritten to claim R9 results.
