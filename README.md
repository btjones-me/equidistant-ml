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
  worker requires Google sign-in before serving any app asset.

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

This runs Python linting and tests, frontend unit and authentication/permission tests, and a
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

### Harrow-to-Sidcup candidate run

The expanded evaluation is isolated under run ID `harrow_sidcup_v1`. It keeps
its graph, 560 origins, 3,538-cell grid, fingerprinted API shards, features,
models, metrics, and atlas separate from the production lineage:

```shell
make expanded-graph
make prepare-expanded-run
make fetch-expanded-data
make train-expanded-candidate
make export-expanded-atlas
make nfr-expanded-candidate
```

`expanded_validate` is a hard gate: training refuses incomplete Cartesian
labels, legacy or mismatched checkpoints, overlapping H3-7 split blocks,
changed inputs, incomplete feature rows, or graph hash drift. The selected model
is trained only from the fresh training split; the tuning split chooses it and
the test split is evaluated once afterward. Candidate outputs remain under
`artifacts/runs/harrow_sidcup_v1/` and are not copied into
`frontend/public/model/` by this workflow.

Atlas export also probes 700 deterministic outer origins using direct model
inference only. Half may nominate adaptive anchors and half remain disjoint for
validation. Extra anchors are accepted only when validation MAE improves without
worsening p90; otherwise the smaller base atlas is retained. Runtime interpolation
uses anchor surface signatures to avoid averaging across sharp transport-access
discontinuities. The training manifest excludes atlas-only parameters from its
lineage hash and records the atlas configuration separately.

Non-paid TravelTime data is for internal evaluation. Confirm a suitable licence
before publishing the candidate or using it commercially.

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

For the isolated Harrow-to-Sidcup candidate, discontinuity-aware interpolation
keeps original-to-original atlas MAE at 3.16 minutes and reduces outer-origin
atlas MAE to 4.92-5.01 minutes. The attempted 100-anchor expansion failed the
disjoint probe gate, so the candidate remains at 765 anchors with no added model
surface payload.

## Deployment

Sites deployment is built from `frontend/`. Google sign-in uses a server-side
OpenID Connect authorization-code flow with PKCE, a browser-bound one-use state,
nonce checks, and verified Google ID tokens. Session cookies are opaque, secure,
HTTP-only, and expire after seven days; logout revokes the stored session.
The old shared password no longer grants access.

Set `AUTH_ORIGIN=https://equidistant.me`, `GOOGLE_CLIENT_ID`,
`GOOGLE_CLIENT_SECRET`, and a random `AUTH_SECRET` of at least 32 characters in
Sites. Register exactly `https://equidistant.me/auth/google/callback` in Google's
web client. `ADMIN_EMAILS` is a comma-separated allowlist of verified Google
email addresses; an empty list grants nobody privileged access. It is evaluated
on each request. All Google accounts may sign in, but only approved accounts
can open diagnostics, view usage aggregates, or call live TravelTime comparisons.
Configure Google's external audience for production and request only
`openid email profile`.

The deployable build must receive a fresh private asset namespace:

```bash
cd frontend
SITE_ASSET_NAMESPACE="_eq_$(openssl rand -hex 32)" npm run build
```

Every public app and model URL passes through the authentication worker before
being mapped to a stored asset. Package `frontend/drizzle/` with the build;
Sites applies the additive account/session/quota migrations before Worker upload.
Generate future migrations with `npx drizzle-kit generate` in `frontend/` and
never rewrite an applied migration.

Participant workspaces remain local to each browser and are keyed by account ID.
Legacy unowned browser data is preserved but is not automatically assigned to
the next person who signs in. Identity-dependent responses are never cached by
shared HTTP caches. The original sharing host redirects to the canonical origin.
Successful sign-ins update anonymous browser-count aggregates and coarse
Cloudflare city/region/country information; raw IPs are not stored.

### Meeting-area recommendations

The review app can research three pubs, restaurants, attractions, or other
destinations around a selected meeting area. Google Places supplies canonical
place details, ratings, map links, and photos; the OpenAI Responses API uses
GPT-5.6 Luna with web search to verify current details and rank the candidates
against the group's request. Both provider keys remain worker-only secrets. Google photos are
proxied through the worker, so neither key is included in browser code or URLs.

For accounts outside `ADMIN_EMAILS`, uncached research attempts are limited to
five in any rolling 60-minute window across devices and sessions. Reservations
are atomic in D1, including simultaneous requests; failed provider attempts also
count. Approved accounts bypass that personal limit. Global caps of 30 per day
and 300 per month still apply to every account. Exact coordinates, area label,
query and model are cached per account for 24 hours; cached results do not consume
another provider call. Photo URLs are signed for the requesting account and
selected photo, with a 24-hour lifetime. Arbitrary paid photo lookups require an
approved account, and existing global photo caps remain in force.
The paid path fails closed if D1 is unavailable, ensuring the application-level
cost controls cannot be bypassed by a storage outage.

Required hosted values are `OPENAI_API_KEY`, `GOOGLE_PLACES_API_KEY`,
`AUTH_ORIGIN`, `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `AUTH_SECRET`,
`RATE_LIMIT_SECRET`, and `ANALYTICS_SECRET`. The Google key
should be API-restricted to Places API (New); it is intentionally not shared
with the existing Directions data-generation key.
