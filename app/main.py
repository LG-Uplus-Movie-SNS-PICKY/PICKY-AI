from fastapi import FastAPI, Depends
from app.database import get_db

app = FastAPI()

# 엔드포인트 정의
@app.get("/")
def root_endpoint():
    return {"message": "Welcome to the Recommender System!"}

@app.get("/check-db")
def check_database(db=Depends(get_db)):
    return {"message": "Database connection successful!"}
