# equidistant-ml


Contains ml for equidistant, an app to estimate journey times.

## Setup

Required: `python dependencies`, `python 3.13`, `uv`, `node`, `npm`

### Install uv

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install python dependencies and editable package

```shell
make install-dev
```

### Run tests

This project uses `pytest` to run tests.

To run the test suite, run:
```
make test
```

This project uses GitHub actions to run the pytest suite on any push automatically.

### Generate Google Directions training data

Create `.env` from the template and add a Google Directions API key:

```shell
make setup
```

```dotenv
GMAPS_API_KEY_MLIGENT=your-api-key
TRAVELTIME_APP_ID=your-traveltime-app-id
TRAVELTIME_API_KEY=your-traveltime-api-key
```

Preview the generated rows without calling Google:

```shell
make generate-data-dry-run
```

Generate a sample parquet dataset:

```shell
make generate-data
```

By default this writes `data/training/directions.<timestamp>.parquet`. Generated
training data and the local request cache are ignored by git. The generator uses
future transit departure times, redacts API keys from saved request URLs, and
caches repeated Google Directions requests locally.

### TravelTime offline surface model

The offline modelling path uses TravelTime's Time Filter Fast matrix endpoint to
label origin-to-grid public-transport surfaces, then trains a local model that
can estimate those surfaces without API calls.

Run the credential-free smoke path:

```shell
make dvc-repro-smoke
```

Generate real TravelTime labels after adding `TRAVELTIME_APP_ID` and
`TRAVELTIME_API_KEY` to `.env`:

```shell
make generate-traveltime-data
make train
make evaluate
```

The production-sized defaults live in `params.yaml`: 500 sampled origins against
a 50x50 destination grid. Generated data, model artifacts, and local DVC cache
are intentionally not committed. The TravelTime fetch stage checkpoints each
origin under `data/interim/traveltime_labels.parts/`, so interrupted runs can be
resumed without refetching completed origins.

### Frontend visualisation

The `frontend/` app is a Vite React TypeScript app intended to become a PWA. It
calls the FastAPI `/api/group-surface` endpoint and renders a heatmap-style grid
for 2-6 friend locations.

```shell
make frontend-install
make frontend-dev
```


## Deployment & CI / CD

This project is deployed on _Heroku_.

Pull requests opened to `main` will trigger a Review App at: https://dashboard.heroku.com/apps/equidistant-ml

Merged PRs to `main` will auto-deploy to `staging`.



[//]: # (## Features)

## TODO
* Create a linear approximator as a baseline model [DONE]
* Some inspiration can be taken from:
https://towardsdatascience.com/simple-example-of-2d-density-plots-in-python-83b83b934f67
* Add tests in GitHub actions CI
## Credits

This package was created with Cookiecutter and the `btjones-me/cookiecutter-pypackage` project template.

* Cookiecutter: https://github.com/cookiecutter/cookiecutter
* btjones-me/cookiecutter-pypackage: https://github.com/btjones-me/cookiecutter-pypackage
