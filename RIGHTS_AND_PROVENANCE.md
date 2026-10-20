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

## R28 update (2026-10-20)

`code/r28/` adds byte-preserved frozen R28 reference and extension sources, original protocols and official Utonia/SPoTr licenses, four H01 result records and three completed retained final checkpoints (H01 local/global, TLP). No in-progress optimizer checkpoints or external pretrained weights are published. `server_export.json` records original file hashes, same-byte extension-to-reference resolution and explicit omissions of datasets, inactive example arrays/images and run logs. Original absolute data/parent-run paths remain provenance prerequisites. `release_manifest.json` covers new publication evidence and navigation documents; historical `code/MANIFEST.sha256.json` is unchanged.

Official direct sources are Utonia (Apache-2.0) and SPoTr (MIT), with their notices preserved. GeoTransformer/Polynormer historical files are inactive in R28. Other citations in the architecture document are explicitly distinguished as design inspiration, technical principles or related comparisons, not copied modules. No new blanket software license is granted. The existing paper draft has not been rewritten to claim R28 results.
