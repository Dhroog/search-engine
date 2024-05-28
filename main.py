from typing import Union
from fastapi import FastAPI
from modules.coreEngine import CoreEngine
from contextlib import asynccontextmanager
from modules.evaluation import Evaluation

quora_engine: CoreEngine | None = None
lotte_engine: CoreEngine | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global quora_engine, lotte_engine
    quora_engine = CoreEngine(25, 0.9,
                              'C://Users//ASUS//Desktop//IR//dataset_ir//quora_docs.csv',
                              'C://Users//ASUS//Desktop//IR//dataset_ir//quora_queries.csv',
                              'C://Users//ASUS//Desktop//IR//dataset_ir//quora_grels.csv')
    lotte_engine = CoreEngine(0.01, 0.9,
                              'C://Users//ASUS//Desktop//IR//dataset_ir//lotte_docs.csv',
                              'C://Users//ASUS//Desktop//IR//dataset_ir//lotte_queries.csv',
                              'C://Users//ASUS//Desktop//IR//dataset_ir//lotte_grels.csv')
    yield
    del quora_engine
    del lotte_engine


app = FastAPI(lifespan=lifespan)


@app.get("/")
def read_hello():
    return {"Hello": "World"}


@app.get("/search")
def read_data(query: Union[str], type: Union[int]):
    if type == 1:
        return quora_engine.search(query)
    else:
        return lotte_engine.search(query)


@app.get("/evaluation")
def read_evaluation():
    eva = Evaluation(quora_engine)
    eva.init_queries_array()
    eva.init_predection_array()
    return {
        'precision_at_k': eva.precision_at_k(),
        'recall_at_k': eva.recall_at_k(),
        'map_at_k': eva.map_at_k(),
        'mrr': eva.mean_reciprocal_rank()
    }
