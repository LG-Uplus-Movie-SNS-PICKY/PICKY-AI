from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db, fetch_user_ratings
from app.recommender import generate_and_store_recommendations, get_user_vector, recommend_movies, movie_embedding_weights, tmdb2idx, idx_to_tmdb
from app.schemas import UserRatings

app = FastAPI()

# 기본 엔드포인트
@app.get("/")
def root_endpoint():
    return {"message": "Welcome to the Picky-AI"}

# 데이터베이스 연결 확인
@app.get("/check-db")
def check_database(db=Depends(get_db)):
    try:
        # 데이터베이스 연결 테스트
        next(get_db())
        return {"message": "Database connection successful!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")

@app.post("/recommendations/all")
def generate_all_recommendations(db: Session = Depends(get_db)):
    try:
        generate_and_store_recommendations(movie_embedding_weights, tmdb2idx, idx_to_tmdb, db, top_k=30)
        return {"message": "Recommendations generated and saved for all users."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate recommendations: {str(e)}")
