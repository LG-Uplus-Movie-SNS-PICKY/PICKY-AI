from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db, fetch_user_ratings
from app.recommender import get_user_vector, recommend_movies, get_movie_details, movie_embedding_weights, tmdb_to_idx, idx_to_tmdb
from app.schemas import UserRatings

app = FastAPI()

# 기본 엔드포인트
@app.get("/")
def root_endpoint():
    return {"message": "Welcome to the Recommender System!"}

# 데이터베이스 연결 확인
@app.get("/check-db")
def check_database(db=Depends(get_db)):
    try:
        # 데이터베이스 연결 테스트
        next(get_db())
        return {"message": "Database connection successful!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")

@app.get("/recommendations/{user_id}")
def get_recommendations(user_id: int, db: Session = Depends(get_db)):
    # 1. 사용자 평점 데이터 가져오기
    user_ratings_dict = fetch_user_ratings(user_id, db)
    if not user_ratings_dict:
        raise HTTPException(status_code=404, detail="User has no ratings.")

    # 2. 사용자 벡터 생성
    user_vector = get_user_vector(user_ratings_dict, movie_embedding_weights, tmdb_to_idx)

    # 3. 추천 영화 생성
    recommended_tmdb_ids = recommend_movies(user_vector, movie_embedding_weights, idx_to_tmdb, top_k=30)

    # 4. DB에서 영화 상세 정보 가져오기
    movie_details = get_movie_details(recommended_tmdb_ids, db)

    return {"user_id": user_id, "recommendations": movie_details}
