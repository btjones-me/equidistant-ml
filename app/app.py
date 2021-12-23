import json
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request, HTTPException
import loguru
from pyprojroot import here


app = FastAPI()

loguru.logger.info('****** Starting service ******')


@app.get("/")
def read_root():
    with open(here() / 'tests/test_data' / 'test_payload.json', mode='r') as f:
        test_payload = json.load(f)
    model_output, _ = payload_to_predictions(test_payload, MODEL_PATH, DF_POI)
    output = {'sample_input': test_payload,
              'calculated_output': model_output}

    return output


@app.post("/")
async def recs(req: Request):

    loguru.logger.info('\n****** New request received ******')
    try:
        payload = await req.json()
        print_payload(payload)
        output, _ = payload_to_predictions(payload, MODEL_PATH, DF_POI)

        # test the output for json compliant-ness (if fails, use default)
        _ = json.dumps(output, allow_nan=False)
        loguru.logger.info('Finished response successfully\n')

    except (TypeError, ValueError, KeyError, ZeroDivisionError, IndexError, AttributeError) as e:
        # attempt to get the aggregate only as a worst case scenario
        # TODO: how to tell backend this happened?
        loguru.logger.exception(e)
        loguru.logger.error(f"ERROR: Failed with error: {e}")
        loguru.logger.error(f"attempting to return default values.")
        output = get_loaded_model(MODEL_PATH, DF_POI)[0].agg_predictions.to_dict()
    except Exception as e:
        loguru.logger.exception(e)
        output = []
        raise HTTPException(status_code=500, detail=f"Something went wrong in service: {e}")

    return output


@app.post("/debug")
async def recs_debug(req: Request):

    loguru.logger.info('\n\n****** New debug request received ******')
    try:
        payload = await req.json()
        loguru.logger.debug(f"{payload=}")

        output = {}

        # load the test (ideal) payload for comparison
        with open(here() / 'tests/test_data' / 'test_payload.json', mode='r') as f:
            test_payload = json.load(f)

        missing_keys = test_payload.keys() - payload.keys()
        output['missing_keys'] = missing_keys if len(missing_keys) > 0 else None
        unexpected_keys = payload.keys() - test_payload.keys()
        output['unexpected_keys'] = unexpected_keys if len(unexpected_keys) > 0 else None

        _, bra = payload_to_predictions(payload, MODEL_PATH, DF_POI)
        output['user_scoresheet'] = bra.user_scoresheet.to_dict()
        return output
    except (TypeError, ValueError, KeyError, ZeroDivisionError, IndexError, AttributeError) as e:
        # attempt to get the aggregate only as a worst case scenario
        loguru.logger.exception(e)
        loguru.logger.error(f"ERROR: Failed with error: {e}\nThis kind of error usually "
                            f"results in default aggregate response.")
        return e
    except Exception as e:
        loguru.logger.exception(e)
        raise HTTPException(status_code=500, detail=f"Something went wrong in service: {e}")


@app.post("/gandalf")
async def gandalf(req: Request):
    json = await req.json()
    gandalf_data = {"hobbits": ["frodo", "sam"]}
    return gandalf_data


if __name__ == "__main__":
    uvicorn.run("router:app", host="0.0.0.0", port=8082)
