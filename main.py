from typing import Union
from fastapi import FastAPI
from modules.coreEngine import CoreEngine
from contextlib import asynccontextmanager
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware

from fastapi.templating import Jinja2Templates

quora_engine: CoreEngine | None = None
lotte_engine: CoreEngine | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global quora_engine, lotte_engine
    quora_engine = CoreEngine(25, 0.9,
                              'C://Users//Asus//Desktop//IR//dataset_ir//quora_docs.csv',
                              'C://Users//Asus//Desktop//IR//dataset_ir//quora_queries.csv',
                              'C://Users//Asus//Desktop//IR//dataset_ir//quora_grels.csv')
    lotte_engine = CoreEngine(25, 0.9,
                              'C://Users//Asus//Desktop//IR//dataset_ir//lotte_docs.csv',
                              'C://Users//Asus//Desktop//IR//dataset_ir//lotte_queries.csv',
                              'C://Users//Asus//Desktop//IR//dataset_ir//lotte_grels.csv')
    yield
    del quora_engine
    del lotte_engine


app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="templates")
origins = ["http://localhost:3000"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def render_template(request: Request):
    return templates.TemplateResponse("search.html", {"request": request, "name": 'yousef'})


@app.get("/search")
def read_data(query: Union[str], type: Union[int]):
    if type == 1:
        return quora_engine.search(query)
    else:
        return lotte_engine.search(query)
