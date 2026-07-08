# equidistant-ml


Contains ml for equidistant, an app to estimate journey times.

## Setup

Required: `python dependencies`, `python 3.8`, `uv`

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
