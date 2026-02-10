# SAE Persona Feature Explorer

Interactive frontend for exploring SAE feature maps built from
`outputs/viz_bundle/*/bundle.json`.

## Run

1. Build a bundle:

```bash
PYTHONPATH=. python pipeline/4_build_viz_bundle.py --config configs/visualization.yaml
```

2. Copy the generated bundle to app public root (or use `VITE_BUNDLE_URL`):

```bash
cp outputs/viz_bundle/general_gemma3_layer40_mean_top5000/bundle.json apps/sae-persona-explorer/public/bundle.json
```

3. Start app:

```bash
npm install
npm run dev
```

## Environment

- `VITE_BUNDLE_URL`: optional URL/path for bundle JSON.
  Default is `/bundle.json`.
