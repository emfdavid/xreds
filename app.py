import logging
import os
import xpublish

from fastapi.middleware.cors import CORSMiddleware

from xreds.spastaticfiles import SPAStaticFiles
#from xreds.dataset_provider import DatasetProvider
from xreds.camus_provider import CamusProvider

logger = logging.getLogger("uvicorn")



gunicorn_logger = logging.getLogger('gunicorn.error')
logger.handlers = gunicorn_logger.handlers
if __name__ != "main":
    logger.setLevel(gunicorn_logger.level)
else:
    logger.setLevel(logging.DEBUG)


rest = xpublish.Rest(
    app_kws=dict(
        title='XREDS', 
        description='XArray Environmental Data Services exposes environmental model data in common data formats for digestion in applications and notebooks',
        openapi_url='/xreds.json'
    ),
    cache_kws=dict(available_bytes=1e9),
    datasets=None
)

rest.register_plugin(CamusProvider())
#rest.register_plugin(DatasetProvider())

app = rest.app

app.add_middleware(
    CORSMiddleware, 
    allow_origins=['*'], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/", SPAStaticFiles(directory="./viewer/dist", html=True), name="viewer")
app.root_path = os.environ.get('ROOT_PATH')


if __name__ == '__main__':
    import uvicorn

    # When run directly, run in debug mode 
    uvicorn.run(
        "app:app", 
        port = 8080,
        reload = True, 
        log_level = 'debug', 
        debug = True
    )