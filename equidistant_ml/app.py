import gzip
import json
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request, HTTPException
import loguru
from pyprojroot import here
from starlette.responses import RedirectResponse, Response

from equidistant_ml.noise.generator import CoordinateDummy

app = FastAPI()

loguru.logger.info('****** Starting service ******')


@app.get("/")
async def read_root():
    loguru.logger.info(f"Received request at /")
    return RedirectResponse(url='/plot/contour')


@app.get("/plot/{plot}")
async def get_plot(plot):
    import io
    from starlette.responses import StreamingResponse

    loguru.logger.info(f"Received request at /plot/{plot}")

    # get img
    c = CoordinateDummy()
    if plot.lower() == "contour":
        fig = c.plot_contour()
    elif plot.lower() == "3d":
        fig = c.plot_3d()
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


@app.post("/predict")
async def recs(req: Request):
    loguru.logger.info('\n****** New request received ******')
    try:
        payload = await req.json()

        # check expected keys exist
        mandatory_keys = {'bbox', 'lat', 'lng', 'x_size', 'y_size'}
        assert all([x in payload.keys() for x in mandatory_keys]), \
            f"Expected key in payload: {mandatory_keys - set(payload.keys())}"

        c = CoordinateDummy(x0=payload["bbox"]["top_left"]["lng"],
                            x1=payload["bbox"]["bottom_right"]["lng"],
                            y0=payload["bbox"]["top_left"]["lat"],
                            y1=payload["bbox"]["bottom_right"]["lat"],
                            x_size=payload["x_size"],
                            y_size=payload["y_size"])

        response = {"lats": c.df.index.to_list(),
                    "lngs": c.df.columns.to_list(),
                    "Z": c.df.values.tolist()}

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
