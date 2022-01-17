import json
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request, HTTPException
import loguru
from pyprojroot import here
from starlette.responses import RedirectResponse

from equidistant_ml.inference.predict import PredictGrid

app = FastAPI()

loguru.logger.info('****** Starting service ******')


@app.get("/")
async def read_root():
    loguru.logger.info(f"Received request at /")
    return RedirectResponse(url='/plot/gaussian/3d')


@app.get("/plot/{mode}/{plot}")
async def get_plot(mode, plot):
    import io
    from starlette.responses import StreamingResponse

    loguru.logger.info(f"Received request at /plot/{mode}/{plot}")
    loguru.logger.info(f'loading default params')

    with open(Path(here() / "tests" / "test_payloads" / "1.json"), mode='r') as f:  # dummy in place
        payload = json.load(f)

    inference_args = unpack_payload(payload)

    # get img
    predictor = PredictGrid(**inference_args)
    if mode.lower() == "linear":
        predictor.get_linear()
    elif mode.lower() == "gaussian":
        predictor.get_gaussian()
    else:
        errs = f"Mode request unrecognised: {mode}"
        loguru.logger.info(errs)
        return errs

    if plot.lower() == "contour":
        fig = predictor.plot_contour()
    elif plot.lower() == "3d":
        fig = predictor.plot_3d()
    else:
        errs = f"Plot request unrecognised: {plot}"
        loguru.logger.info(errs)
        return errs
    loguru.logger.info("Fig generated")

    # save img
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    loguru.logger.info("Fig saved to file buf")
    buf.seek(0)
    loguru.logger.info("Buffer sought, returning")

    return StreamingResponse(buf, media_type="image/png")


@app.post("/debug")
async def debug(req: Request):
    loguru.logger.info('\n****** New request received ******')
    payload = await req.json()
    output = {'you sent me': payload,
              'you got back': "ffs ben hurry up and implement me already"}
    return output


def unpack_payload(payload):
    # check expected keys exist
    mandatory_keys = {'bbox', 'lat', 'lng', 'x_size', 'y_size'}
    assert all([x in payload.keys() for x in mandatory_keys]), \
        f"Expected key in payload: {mandatory_keys - set(payload.keys())}"

    # get payload args - ignores extra args
    inference_args = {
        "x0": payload["bbox"]["top_left"]["lng"],
        "x1": payload["bbox"]["bottom_right"]["lng"],
        "y0": payload["bbox"]["top_left"]["lat"],
        "y1": payload["bbox"]["bottom_right"]["lat"],
        "x_size": payload["x_size"],
        "y_size": payload["y_size"],
        "lat": payload["lat"],
        "lng": payload["lng"]
    }

    return inference_args


def prep_response(preds_df):
    """Format the dataframe for returning."""
    return {"lats": preds_df.index.to_list(),
            "lngs": preds_df.columns.to_list(),
            "Z": preds_df.values.tolist()}


@app.post("/predict")
async def recs(req: Request):
    loguru.logger.info('\n****** New request received ******')
    try:
        payload = await req.json()

        inference_args = {
            "x0": payload["bbox"]["top_left"]["lng"],
            "x1": payload["bbox"]["bottom_right"]["lng"],
            "y0": payload["bbox"]["top_left"]["lat"],
            "y1": payload["bbox"]["bottom_right"]["lat"],
            "x_size": payload["x_size"],
            "y_size": payload["y_size"]
        }
        user_location_args = {
            "lat": payload["lat"],
            "lng": payload["lng"]
        }

        if payload['mode'].lower() == 'walking':
            # use linear approximator
            loguru.logger.info(f"mode: {payload['mode']}, engine: 'linear'")
            predictor = PredictGrid(**inference_args, **user_location_args)
            df = predictor.get_linear()
            response = prep_response(df)
        elif payload['mode'].lower() == 'gaussian':
            # use gaussian approximator
            loguru.logger.info(f"mode: {payload['mode']}, engine: 'gaussian'")
            predictor = PredictGrid(**inference_args, **user_location_args)
            df = predictor.get_linear()
            response = prep_response(df)
        else:  # default to linear
            loguru.logger.info(f"mode unknown: {payload['mode']}, engine: 'default to linear'")
            predictor = PredictGrid(**inference_args, **user_location_args)
            df = predictor.get_linear()
            response = prep_response(df)

        # compression
        # response = gzip.compress(json.dumps(response).encode('utf-8'))
        # response = Response(response, media_type='application/gzip')

    except (TypeError, ValueError, KeyError, ZeroDivisionError, IndexError, AttributeError, AssertionError) as e:
        # TODO: how to tell backend this happened?
        loguru.logger.exception(e)
        loguru.logger.error(f"ERROR: Failed with error: {e}")
        raise HTTPException(status_code=500, detail=f"Something went wrong in service: {e}")
    except Exception as e:
        loguru.logger.exception(e)
        raise HTTPException(status_code=500, detail=f"Something went wrong in service: {e}")

    return response


if __name__ == "__main__":
    uvicorn.run("router:app", host="0.0.0.0", port=8082)
