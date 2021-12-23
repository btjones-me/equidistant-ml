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
    c = CoordinateDummy(size=1000)
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
    output = {}
    try:
        payload = await req.json()
        with open(Path(here() / "tests" / "test_payloads" / "1.json"), mode='r') as f:  # dummy in place
            payload = json.load(f)

        for key in payload:
            c = CoordinateDummy(size=100)
            output[key] = c.df.to_json()
        assert len(output) == len(payload)

        compress = False
        if compress:
            output = gzip.compress(json.dumps(output).encode('utf-8'))
            response = Response(output, media_type='application/gzip')

        response = output

    except (TypeError, ValueError, KeyError, ZeroDivisionError, IndexError, AttributeError, AssertionError) as e:
        # TODO: how to tell backend this happened?
        loguru.logger.exception(e)
        loguru.logger.error(f"ERROR: Failed with error: {e}")
        response = [f"Internal Python exception. {e}"]
        raise HTTPException(status_code=500, detail=f"Something went wrong in service: {e}")
    except Exception as e:
        loguru.logger.exception(e)
        response = [f"Internal Python exception. {e}"]
        raise HTTPException(status_code=500, detail=f"Something went wrong in service: {e}")

    return response


if __name__ == "__main__":
    uvicorn.run("router:app", host="0.0.0.0", port=8082)
