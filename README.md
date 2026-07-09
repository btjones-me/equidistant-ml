# Equidistant

Equidistant finds fair meeting areas for groups of friends using public-transport
travel time rather than straight-line distance. The current product covers
central London and Zones 1-3, supports groups of 2-6, and runs its primary
surface model offline.

## Product

The frontend has two experiences backed by one persisted workspace:

- **Equidistant** is the production map. Add or drag starting points, choose a
  group objective, and inspect suggested meeting areas and per-person journeys.
- **Developer mode** exposes coverage, resolution, colour controls, individual
  model/graph layers, TravelTime references, signed error, and cell diagnostics.

Participant selections, coordinates, scoring strategy, coverage, palette,
custom colour stops, and colour range are shared between both modes and saved
locally in the browser.

## Architecture

- `equidistant_ml/` contains the FastAPI development API, feature engineering,
  transport graph, model training, evaluation, and TravelTime integration.
- `frontend/` is a Vite React TypeScript application using Leaflet and H3 cells.
- `frontend/public/model/` is a quantised browser atlas generated from the best
  local model. It keeps the hosted app dynamic without shipping Python, model
  credentials, or a live TravelTime dependency.
- `dvc.yaml` and `params.yaml` define the reproducible data and training stages.
- `.openai/hosting.json` and `frontend/worker/` package the app for Sites. The
  worker enforces the shared-password gate before serving any app asset.

The browser atlas interpolates from 560 origin anchors over 3,032 mixed-priority
H3 destinations. Its measured interpolation MAE against direct local-model
inference is stored in `frontend/public/model/atlas.json`.

## Local setup

Required: Python 3.13, [uv](https://docs.astral.sh/uv/), Node.js, and npm.

```shell
make install-dev
make frontend-install
```

Create local credentials when generating or validating TravelTime labels:

```shell
make setup
```

```dotenv
GMAPS_API_KEY_MLIGENT=your-google-api-key
TRAVELTIME_APP_ID=your-traveltime-app-id
TRAVELTIME_API_KEY=your-traveltime-api-key
```

Start the development API and frontend in separate terminals:

```shell
make run-server
make frontend-dev
```

Open `http://localhost:5173/`. The production view uses the browser atlas; the
local API automatically powers richer Developer mode responses and live/cached
TravelTime comparisons.

## Quality checks

```shell
make check
```

This runs Python linting and tests, frontend unit and password-gate tests, and a
production/Sites build. Individual commands remain available:

```shell
make test
make lint
make frontend-test
make frontend-build
```

## Model pipeline

Run the credential-free model and graph smoke paths:

```shell
make dvc-repro-smoke
make dvc-repro-graph-smoke
```

Generate real TravelTime labels and train/evaluate the local models after adding
credentials to `.env`:

```shell
make generate-traveltime-data
make fetch-transport-data
make build-transport-graph
make train
make evaluate
make evaluate-corridors
```

The TravelTime fetch checkpoints each origin under
`data/interim/traveltime_labels.parts/`, allowing interrupted runs to resume
without repeating completed API calls. Generated data, local models, caches, and
experiment reports are intentionally excluded from git.

The graph layer uses TfL topology plus deterministic rail-corridor fallbacks for
Tube, Overground, Elizabeth line, Thameslink, and National Rail. TravelTime
remains the label source; the graph is a topology prior and diagnostic baseline.

## Browser atlas

Regenerate the production inference atlas after promoting a new model:

```shell
make export-browser-atlas
```

The exporter writes compact `uint8` model and graph surfaces plus H3 metadata,
then validates interpolation against direct model inference. Commit all three
files in `frontend/public/model/` together.

## Current evidence

The promoted graph-augmented model records approximately 2.70 minutes MAE and
5.87 minutes p90 absolute error on the central holdout. The production atlas
adds approximately 1.39 minutes MAE relative to direct model inference. These
figures describe weekday-morning public-transport estimates, not guarantees for
a specific journey.

## Deployment

Sites deployment is built from `frontend/`. `SITE_PASSWORD` is supplied as a
hosted environment value; it is never stored in source or bundled assets. The
deployable build should also receive a fresh private asset namespace:

```bash
cd frontend
SITE_ASSET_NAMESPACE="_eq_$(openssl rand -hex 32)" npm run build
```

That namespace ensures every public app and model URL reaches the password
worker before it is mapped to a stored asset. The static offline atlas then lets
the production experience serve a small group of concurrent users without a
Python service or per-request API cost.
