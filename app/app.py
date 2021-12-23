import json
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request, HTTPException
import loguru
from pyprojroot import here

from equidistant_ml.noise.generator import CoordinateDummy

app = FastAPI()

loguru.logger.info('****** Starting service ******')


@app.get("/")
def read_root(req: Request):
    payload = await req.json()
    output = {'you sent me': payload,
              'you got back': "ffs ben hurry up and implement me already"}
    return output


@app.post("/predict")
async def recs(req: Request):

    loguru.logger.info('\n****** New request received ******')
    try:
        payload = await req.json()
        with open(Path(here() / "tests" / "test_payloads" / "1.json"), mode='r') as f:  # dummy in place
            payload = json.load(f)

        output = {}
        for key in payload:
            c = CoordinateDummy(size=1000)
            output[key] = c.df.to_json()

        assert len(output) == len(payload)
    except (TypeError, ValueError, KeyError, ZeroDivisionError, IndexError, AttributeError, AssertionError) as e:
        # TODO: how to tell backend this happened?
        loguru.logger.exception(e)
        loguru.logger.error(f"ERROR: Failed with error: {e}")
        output = [f"Internal Python exception. {e}"]
        raise HTTPException(status_code=500, detail=f"Something went wrong in service: {e}")
    except Exception as e:
        loguru.logger.exception(e)
        output = [f"Internal Python exception. {e}"]
        raise HTTPException(status_code=500, detail=f"Something went wrong in service: {e}")

    return output


if __name__ == "__main__":
    uvicorn.run("router:app", host="0.0.0.0", port=8082)
