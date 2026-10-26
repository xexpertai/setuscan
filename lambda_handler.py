"""AWS Lambda entry point: the same FastAPI app, behind API Gateway (HTTP API)."""
from mangum import Mangum

from setuscan.api import app

handler = Mangum(app, lifespan="off")
