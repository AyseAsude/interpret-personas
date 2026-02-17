# SAE Persona Feature Explorer

Interactive frontend for:
- SAE feature maps built from
`outputs/viz_bundle/*/bundle.json`.
- Persona drift runs built by `notebooks/persona_drift.ipynb`.

## Run

1. Build bundle(s):

```bash
PYTHONPATH=. python pipeline/4_build_viz_bundle.py --config configs/visualization.yaml
```

For centering toggle support in UI, build two bundles with different dataset names:
- non-centered: `question_centering: false`
- centered: `question_centering: true`

2. Provide bundle files to the app.

```bash
cp outputs/viz_bundle/general_gemma3_layer40_mean_top2000/bundle.json \
  apps/sae-persona-explorer/public/bundle_non_centered.json

cp outputs/viz_bundle/general_gemma3_layer40_mean_top2000_centered/bundle.json \
  apps/sae-persona-explorer/public/bundle_centered.json
```

3. Start app:

```bash
npm install
npm run dev
```

Routes:
- SAE explorer (default): `/#/`
- Persona drift explorer: `/#/persona-drift`

4. Provide Persona Drift files (optional).

Copy run files into `apps/sae-persona-explorer/public/persona-drift/`.
For multi-run mode, provide `runs_manifest.json` in that folder.
Each run entry should include:
- `conversation_name` (used as dropdown label)
- `ui_bundle_url`
- optional `conversation_url`
- optional `sensitive: true` (or use `conversation_name: "Self-harm"` for automatic warning styling)

## Environment

- `VITE_BUNDLE_URL`: optional URL/path for bundle JSON.
  Default is `/bundle.json`.
- `VITE_BUNDLE_URL_NON_CENTERED`: optional URL/path for non-centered bundle.
  Falls back to `VITE_BUNDLE_URL` (or `/bundle.json`).
- `VITE_BUNDLE_URL_CENTERED`: optional URL/path for centered bundle.
  When set, the Settings tab can switch Question Centering ON/OFF.
- `VITE_PERSONA_DRIFT_URL`: optional URL/path for the "Open Persona Drift Explorer" nav link.
  Default is `#/persona-drift`.
- `VITE_PERSONA_EXPLORER_URL`: optional URL/path for the drift view "Back to SAE Persona Explorer" link.
  Default is `#/`.
- `VITE_GITHUB_PROFILE_URL`: optional GitHub URL used by the topbar icon in both views.
  Default is `https://github.com/AyseAsude/interpret-personas`.
- `VITE_DRIFT_UI_BUNDLE_URL`: optional URL/path for drift `ui_bundle.json`.
  Default is `${BASE_URL}persona-drift/ui_bundle.json`.
- `VITE_DRIFT_CONVERSATION_URL`: optional URL/path for drift `conversation.json`.
  Default is `${BASE_URL}persona-drift/conversation.json`.
- `VITE_DRIFT_RUNS_MANIFEST_URL`: optional URL/path for drift `runs_manifest.json`.
  Default is `${BASE_URL}persona-drift/runs_manifest.json`.
- `VITE_DRIFT_NEURONPEDIA_ID`: optional Neuronpedia ID for drift feature links
  (for example `gemma-3-27b-it/40-gemmascope-2-res-65k`).
  When unset, the app tries to infer it from drift `run_meta`.

Example:

```bash
VITE_BUNDLE_URL_NON_CENTERED=/bundle_non_centered.json \
VITE_BUNDLE_URL_CENTERED=/bundle_centered.json \
npm run dev
```
