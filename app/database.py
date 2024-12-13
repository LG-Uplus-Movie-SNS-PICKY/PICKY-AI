import json
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from typing import Dict, List

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
engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

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

def fetch_movie_details(tmdb_ids: list[int], db) -> list[dict]:
    """
    TMDb ID 리스트를 받아 데이터베이스에서 영화 상세 정보를 가져옵니다.
    """
    if not tmdb_ids:
        return []  # TMDb ID가 없는 경우 빈 리스트 반환

    # SQL 쿼리 작성
    query = text("""
        SELECT id, title, poster_url
        FROM movie
        WHERE id IN :movie_ids
    """)

    # TMDb ID를 Python 기본 정수로 변환
    tmdb_ids = [int(tmdb_id) for tmdb_id in tmdb_ids]

    # 쿼리 실행 및 결과 가져오기
    result = db.execute(query, {"movie_ids": tuple(tmdb_ids)}).fetchall()

    # 결과 가공
    movie_details = []
    for row in result:
        movie = {
            "id": row.id,
            "title": row.title,
            "poster_url": row.poster_url,
        }
        movie_details.append(movie)

    return movie_details
