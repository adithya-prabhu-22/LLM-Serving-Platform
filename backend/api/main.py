from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import UploadFile
from fastapi import File
from fastapi import Form

from backend.api.routes.health import (
    get_health,
)

from backend.api.routes.root import (
    root,
)

from backend.api.routes.models import (
    get_models,
    get_model_by_id,
    get_model_status,
    build_model_route,
)

from backend.api.routes.generation import (
    generate_text,
)

from backend.api.routes.admin import (
    upload_model_route,
    delete_model_route,
)

from backend.api.schemas.generate_request import (
    GenerateRequest,
)

from backend.api.schemas.generate_response import (
    GenerateResponse,
)

from backend.api.schemas.upload_model_response import (
    UploadModelResponse,
)

from backend.services.validator import (
    validate_config_file,
    validate_weights_file,
    validate_tokenizer_input,
)


app = FastAPI(
    title="LLM Serving Platform",
    version="1.0.0",
)


@app.get("/")
def root_route():

    return root()


@app.get("/models")
def models():

    return get_models()


@app.get(
    "/models/{model_id}"
)
def model_by_id(
    model_id: str,
):

    try:

        return get_model_by_id(
            model_id
        )

    except ValueError as error:

        raise HTTPException(
            status_code=404,
            detail=str(error),
        )


@app.get(
    "/models/status/{model_id}"
)
def model_status(
    model_id: str,
):

    try:

        return get_model_status(
            model_id
        )

    except ValueError as error:

        raise HTTPException(
            status_code=404,
            detail=str(error),
        )


@app.post(
    "/models/build/{model_id}"
)
def build_model(
    model_id: str,
):

    try:

        return build_model_route(
            model_id
        )

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error),
        )


@app.post(
    "/generate",
    response_model=GenerateResponse,
)
def generate_api(
    request: GenerateRequest,
):

    try:

        return generate_text(
            model_id=request.model_id,
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
        )

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error),
        )


@app.post(
    "/admin/models/upload",
    response_model=UploadModelResponse,
)
async def upload_model_api(

    model_id: str = Form(...),

    name: str = Form(...),

    architecture: str = Form(...),

    config_file: UploadFile = File(...),

    weights_file: UploadFile = File(...),

    tokenizer_file: UploadFile = File(...),
):

    try:

        validate_config_file(
            config_file.filename
        )

        validate_weights_file(
            weights_file.filename
        )

        validate_tokenizer_input(
            tokenizer_file=tokenizer_file.filename,
        )

        return upload_model_route(

            model_id=model_id,

            name=name,

            architecture=architecture,

            config_content=await config_file.read(),

            weights_content=await weights_file.read(),

            tokenizer_content=await tokenizer_file.read(),
        )

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error),
        )


@app.delete(
    "/admin/models/{model_id}"
)
def delete_model_api(
    model_id: str,
):

    try:

        return delete_model_route(
            model_id
        )

    except ValueError as error:

        raise HTTPException(
            status_code=404,
            detail=str(error),
        )
        
@app.get(
    "/health"
)
def health():

    return get_health()