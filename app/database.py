import json
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from typing import Dict, List
import pickle

# JSON 파일에서 DB 설정 로드
def load_db_config():
    try:
        with open("config/db_config.json", "r") as file:
            config = json.load(file)
        return config["database"]
    except FileNotFoundError:
        raise Exception("DB 설정 파일이 없습니다. 'config/db_config.json'을 확인하세요.")
    except json.JSONDecodeError:
        raise Exception("DB 설정 파일의 JSON 형식이 잘못되었습니다.")

# 데이터베이스 URL 생성
db_config = load_db_config()
DATABASE_URL = f"mysql+pymysql://{db_config['username']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['db_name']}"

# SQLAlchemy 세션 설정
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def fetch_all_users(db):
    """
    데이터베이스에서 모든 사용자 ID를 가져옵니다.
    db: 데이터베이스 연결 객체
    """
    query = text("""
        SELECT id
        FROM user
    """)
    result = db.execute(query).fetchall()  # result는 튜플 리스트로 반환됩니다
    users = []
    for row in result:
        users.append({
            "user_id": row[0],  # row[0]으로 접근
            "ratings": fetch_user_ratings(row[0], db)  # 사용자별 평점 데이터 가져오기
        })
    return users

 

def fetch_user_ratings(user_id: int, db) -> Dict[int, float]:
    """
    특정 사용자(user_id)의 평점 데이터를 가져옵니다.
    """
    query = text("""
        SELECT movie_id, rating
        FROM line_review
        WHERE user_id = :user_id
    """)
    result = db.execute(query, {"user_id": user_id}).fetchall()
    # 영화 ID와 평점을 딕셔너리로 반환
    return {row.movie_id: row.rating for row in result}

def fetch_movie_details(tmdb_ids, db):
    """
    TMDb ID를 기반으로 영화 세부 정보를 가져옵니다.
    tmdb_ids: TMDb ID 리스트
    db: 데이터베이스 연결 객체
    """
    tmdb_ids = [int(id) for id in tmdb_ids]
    if not tmdb_ids:
        return []

    query = text("""
        SELECT id, title, popularity  -- 인기도 추가
        FROM movie
        WHERE id IN :movie_ids
    """)
    

    try:
        result = db.execute(query, {"movie_ids": tuple(tmdb_ids)}).fetchall()
        movie_details = []
        for row in result:
            movie_details.append({
                "id": row.id,
                "title": row.title,
                "popularity": row.popularity,  # 인기도 추가
            })
        return movie_details
    except Exception as e:
        print(f"Error fetching movie details: {e}")
        return []


def fetch_user_liked_movies(user_id: int, db) -> list[int]:

    query = text("""
        SELECT movie_id
        FROM movie_like
        WHERE user_id = :user_id
    """)
    result = db.execute(query, {"user_id": user_id}).fetchall()
    return [row.movie_id for row in result]


def save_user_recommendations(recommendations, db):
    """
    사용자 ID와 각 추천 영화 ID 및 유사도를 개별 행으로 저장합니다.
    recommendations: [{"user_id": 1, "recommended_movies": [{"movie_id": 101, "similarity": 0.9}, ...]}, ...]
    db: SQLAlchemy Session 객체
    """
    query = text("""
        INSERT INTO recommend (user_id, movie_id, similarity, created_at, updated_at)
        VALUES (:user_id, :movie_id, :similarity, NOW(), NOW())
        ON DUPLICATE KEY UPDATE
        similarity = :similarity,
        updated_at = NOW()
    """)

    try:
        for rec in recommendations:
            user_id = rec["user_id"]
            recommended_movies = rec.get("recommended_movies", [])

            if not recommended_movies:
                print(f"No recommendations to save for user {user_id}.")
                continue

            for movie in recommended_movies:
                db.execute(query, {
                    "user_id": user_id,
                    "movie_id": movie["movie_id"],
                    "similarity": movie["similarity"]  # similarity 값 저장
                })
            db.commit()
    except Exception as e:
        db.rollback()
        raise Exception(f"Error saving recommendations for user {user_id}: {e}")
