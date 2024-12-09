from fastapi import FastAPI
from app.recommender import recommend

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Recommender System!"}

@app.post("/recommend/")
async def get_recommendation(user_id: int, movie_id: int, occupation: int):
    return recommend(user_id, movie_id, occupation)
