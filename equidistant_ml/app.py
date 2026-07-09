import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

import loguru
import requests
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field
from pyprojroot import here

from equidistant_ml.inference.predict import PredictGrid
from equidistant_ml.surfaces.predict import (
    predict_group_surface,
    predict_origin_surface,
    reference_group_surface,
    surface_to_grid_response,
)

app = FastAPI(title="Equidistant API", version="0.2.0")
app.add_middleware(GZipMiddleware, minimum_size=1000, compresslevel=5)

loguru.logger.info("****** Starting service ******")


class FriendLocation(BaseModel):
    lat: float
    lng: float
    name: str | None = None


class SurfaceRequest(BaseModel):
    origin: FriendLocation
    x_size: int | None = Field(default=None, ge=5, le=100)
    y_size: int | None = Field(default=None, ge=5, le=100)
    grid_mode: Literal["uniform", "h3"] = "uniform"
    focus: Literal["central", "inner", "wide"] | None = None
    detail: Literal["fast", "fine"] | None = None


class GroupSurfaceRequest(BaseModel):
    friends: list[FriendLocation] = Field(min_length=2, max_length=6)
    combine: Literal["max", "mean", "fairness", "balanced"] = "balanced"
    included_friend_indexes: list[int] | None = None
    x_size: int | None = Field(default=None, ge=5, le=100)
    y_size: int | None = Field(default=None, ge=5, le=100)
    grid_mode: Literal["uniform", "h3"] = "uniform"
    focus: Literal["central", "inner", "wide"] | None = None
    detail: Literal["fast", "fine"] | None = None


class ComparisonSurfaceRequest(GroupSurfaceRequest):
    grid_mode: Literal["h3"] = "h3"


@app.get("/")
def read_root():
    return {
        "service": "equidistant-api",
        "status": "ok",
        "docs": "/docs",
        "frontend": "Run the Vite app from frontend/",
    }


@app.get("/plot/{mode}/{plot}", deprecated=True)
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


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "equidistant-api",
        "model": "offline-public-transport",
    }


@app.post("/api/surface")
def surface(payload: SurfaceRequest):
    try:
        df = predict_origin_surface(
            payload.origin.lat,
            payload.origin.lng,
            x_size=payload.x_size,
            y_size=payload.y_size,
            grid_mode=payload.grid_mode,
            focus=payload.focus,
            detail=payload.detail,
        )
        return surface_to_grid_response(df, "travel_time_minutes")
    except Exception as e:
        loguru.logger.exception(e)
        raise HTTPException(status_code=500, detail=f"Surface prediction failed: {e}")


@lru_cache(maxsize=32)
def _cached_group_surface(payload_json: str) -> dict:
    payload = GroupSurfaceRequest.model_validate_json(payload_json)
    friends = [friend.model_dump() for friend in payload.friends]
    frame = predict_group_surface(
        friends,
        combine=payload.combine,
        included_friend_indexes=payload.included_friend_indexes,
        x_size=payload.x_size,
        y_size=payload.y_size,
        grid_mode=payload.grid_mode,
        focus=payload.focus,
        detail=payload.detail,
    )
    return surface_to_grid_response(frame, "model_score_minutes")


@app.post("/api/group-surface")
def group_surface(payload: GroupSurfaceRequest):
    try:
        return _cached_group_surface(payload.model_dump_json())
    except Exception as e:
        loguru.logger.exception(e)
        raise HTTPException(
            status_code=500,
            detail=f"Group surface prediction failed: {e}",
        )


@app.post("/api/comparison-surface")
def comparison_surface(payload: ComparisonSurfaceRequest):
    try:
        friends = [friend.model_dump() for friend in payload.friends]
        df, metrics = reference_group_surface(
            friends,
            combine=payload.combine,
            included_friend_indexes=payload.included_friend_indexes,
            x_size=payload.x_size,
            y_size=payload.y_size,
            grid_mode=payload.grid_mode,
            focus=payload.focus,
            detail=payload.detail,
        )
        response = surface_to_grid_response(df, "model_score_minutes")
        response["metadata"]["comparison"] = metrics
        response["metadata"]["value_columns"] = {
            "model": "model_score_minutes",
            "graph": "graph_score_minutes",
            "residual": "model_residual_minutes",
            "reference": "reference_score_minutes",
            "error": "signed_error_minutes",
        }
        return response
    except Exception as e:
        loguru.logger.exception(e)
        raise HTTPException(
            status_code=500,
            detail=f"TravelTime comparison failed: {e}",
        )


@lru_cache(maxsize=128)
def _geocode_london(query: str) -> tuple[dict, ...]:
    response = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={
            "q": f"{query}, London, UK",
            "format": "jsonv2",
            "limit": "5",
            "countrycodes": "gb",
            "viewbox": "-0.60,51.75,0.35,51.20",
            "bounded": "1",
            "addressdetails": "1",
        },
        headers={"User-Agent": "equidistant-ml/0.2 (github.com/btjones-me)"},
        timeout=8,
    )
    response.raise_for_status()
    results = []
    for item in response.json():
        address = item.get("address") or {}
        name = (
            item.get("name")
            or address.get("suburb")
            or address.get("neighbourhood")
            or address.get("road")
            or str(item.get("display_name", "London")).split(",", 1)[0]
        )
        results.append(
            {
                "name": str(name),
                "lat": float(item["lat"]),
                "lng": float(item["lon"]),
                "detail": str(item.get("display_name", "")),
            }
        )
    return tuple(results)


@app.get("/api/geocode")
def geocode(q: str = Query(min_length=2, max_length=120)):
    try:
        return {"results": list(_geocode_london(q.strip()))}
    except requests.RequestException as error:
        loguru.logger.warning("Location search failed: {}", error)
        raise HTTPException(status_code=503, detail="Location search unavailable")


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


@app.post("/predict", deprecated=True)
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
            df = predictor.get_gaussian()
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
