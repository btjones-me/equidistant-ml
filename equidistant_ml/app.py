import json
from pathlib import Path
from typing import Literal

import loguru
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from pyprojroot import here
from starlette.responses import RedirectResponse

from equidistant_ml.inference.predict import PredictGrid
from equidistant_ml.surfaces.predict import (
    predict_group_surface,
    predict_origin_surface,
    surface_to_grid_response,
)

app = FastAPI()

loguru.logger.info("****** Starting service ******")


class FriendLocation(BaseModel):
    lat: float
    lng: float
    name: str | None = None


class SurfaceRequest(BaseModel):
    origin: FriendLocation
    x_size: int | None = Field(default=None, ge=5, le=100)
    y_size: int | None = Field(default=None, ge=5, le=100)


class GroupSurfaceRequest(BaseModel):
    friends: list[FriendLocation] = Field(min_length=2, max_length=6)
    combine: Literal["max", "mean", "fairness", "balanced"] = "balanced"
    x_size: int | None = Field(default=None, ge=5, le=100)
    y_size: int | None = Field(default=None, ge=5, le=100)


@app.get("/")
async def read_root():
    loguru.logger.info("Received request at /")
    return RedirectResponse(url="/plot/gaussian/3d")


@app.get("/plot/{mode}/{plot}")
async def get_plot(mode, plot):
    import io

    from starlette.responses import StreamingResponse

    loguru.logger.info(f"Received request at /plot/{mode}/{plot}")
    loguru.logger.info("loading default params")

    with open(
        Path(here() / "tests" / "test_payloads" / "1.json"), mode="r"
    ) as f:  # dummy in place
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
    loguru.logger.info("\n****** New request received ******")
    payload = await req.json()
    output = {
        "you sent me": payload,
        "you got back": "ffs ben hurry up and implement me already",
    }
    return output


@app.post("/api/surface")
async def surface(payload: SurfaceRequest):
    try:
        df = predict_origin_surface(
            payload.origin.lat,
            payload.origin.lng,
            x_size=payload.x_size,
            y_size=payload.y_size,
        )
        return surface_to_grid_response(df, "travel_time_minutes")
    except Exception as e:
        loguru.logger.exception(e)
        raise HTTPException(status_code=500, detail=f"Surface prediction failed: {e}")


@app.post("/api/group-surface")
async def group_surface(payload: GroupSurfaceRequest):
    try:
        friends = [friend.model_dump() for friend in payload.friends]
        df = predict_group_surface(
            friends,
            combine=payload.combine,
            x_size=payload.x_size,
            y_size=payload.y_size,
        )
        return surface_to_grid_response(df, "score_minutes")
    except Exception as e:
        loguru.logger.exception(e)
        raise HTTPException(
            status_code=500,
            detail=f"Group surface prediction failed: {e}",
        )


def unpack_payload(payload):
    # check expected keys exist
    mandatory_keys = {"bbox", "lat", "lng", "x_size", "y_size"}
    missing_keys = mandatory_keys - set(payload.keys())
    if missing_keys:
        raise ValueError(f"Expected key in payload: {missing_keys}")

    # get payload args - ignores extra args
    inference_args = {
        "x0": payload["bbox"]["top_left"]["lng"],
        "x1": payload["bbox"]["bottom_right"]["lng"],
        "y0": payload["bbox"]["top_left"]["lat"],
        "y1": payload["bbox"]["bottom_right"]["lat"],
        "x_size": payload["x_size"],
        "y_size": payload["y_size"],
        "lat": payload["lat"],
        "lng": payload["lng"],
    }

    return inference_args


def prep_response(preds_df):
    """Format the dataframe for returning."""
    return {
        "lats": preds_df.index.to_list(),
        "lngs": preds_df.columns.to_list(),
        "Z": preds_df.values.tolist(),
    }


@app.post("/predict")
async def recs(req: Request):
    loguru.logger.info("\n****** New request received ******")
    try:
        payload = await req.json()

        inference_args = {
            "x0": payload["bbox"]["top_left"]["lng"],
            "x1": payload["bbox"]["bottom_right"]["lng"],
            "y0": payload["bbox"]["top_left"]["lat"],
            "y1": payload["bbox"]["bottom_right"]["lat"],
            "x_size": payload["x_size"],
            "y_size": payload["y_size"],
        }
        user_location_args = {"lat": payload["lat"], "lng": payload["lng"]}

        if payload["mode"].lower() == "walking":
            # use linear approximator
            loguru.logger.info(f"mode: {payload['mode']}, engine: 'linear'")
            predictor = PredictGrid(**inference_args, **user_location_args)
            df = predictor.get_linear()
            response = prep_response(df)
        elif payload["mode"].lower() == "gaussian":
            # use gaussian approximator
            loguru.logger.info(f"mode: {payload['mode']}, engine: 'gaussian'")
            predictor = PredictGrid(**inference_args, **user_location_args)
            df = predictor.get_linear()
            response = prep_response(df)
        else:  # default to linear
            loguru.logger.info(
                f"mode unknown: {payload['mode']}, engine: 'default to linear'"
            )
            predictor = PredictGrid(**inference_args, **user_location_args)
            df = predictor.get_linear()
            response = prep_response(df)

        # compression
        # response = gzip.compress(json.dumps(response).encode('utf-8'))
        # response = Response(response, media_type='application/gzip')

    except (
        TypeError,
        ValueError,
        KeyError,
        ZeroDivisionError,
        IndexError,
        AttributeError,
        AssertionError,
    ) as e:
        # TODO: how to tell backend this happened?
        loguru.logger.exception(e)
        loguru.logger.error(f"ERROR: Failed with error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Something went wrong in service: {e}"
        )
    except Exception as e:
        loguru.logger.exception(e)
        raise HTTPException(
            status_code=500, detail=f"Something went wrong in service: {e}"
        )

    return response


if __name__ == "__main__":
    uvicorn.run("equidistant_ml.app:app", host="127.0.0.1", port=8082)
