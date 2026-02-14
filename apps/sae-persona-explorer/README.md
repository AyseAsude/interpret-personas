# SAE Persona Feature Explorer

Interactive frontend for exploring SAE feature maps built from
`outputs/viz_bundle/*/bundle.json`.

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

## Environment

- `VITE_BUNDLE_URL`: optional URL/path for bundle JSON.
  Default is `/bundle.json`.
- `VITE_BUNDLE_URL_NON_CENTERED`: optional URL/path for non-centered bundle.
  Falls back to `VITE_BUNDLE_URL` (or `/bundle.json`).
- `VITE_BUNDLE_URL_CENTERED`: optional URL/path for centered bundle.
  When set, the Settings tab can switch Question Centering ON/OFF.

Example:

```bash
VITE_BUNDLE_URL_NON_CENTERED=/bundle_non_centered.json \
VITE_BUNDLE_URL_CENTERED=/bundle_centered.json \
npm run dev
```
