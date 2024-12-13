import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy.sql import text  # SQL 텍스트 명시적으로 선언
from app.database import get_db  # 기존 데이터베이스 연결 설정

def export_data_to_csv():
    # 데이터베이스 연결 가져오기
    db: Session = next(get_db())

    # SQL 쿼리 작성
    query = text("""
    SELECT
        lr.user_id,
        lr.movie_id,
        lr.rating,
        u.gender,
        TIMESTAMPDIFF(YEAR, u.birthdate, CURDATE()) AS age
    FROM
        line_review lr
    JOIN
        user u
    ON
        lr.user_id = u.id
    WHERE
        lr.is_deleted = 0;
    """)

    # SQLAlchemy의 connection.execute()를 사용하여 쿼리 실행
    result = db.execute(query)

    # 결과를 DataFrame으로 변환
    data = pd.DataFrame(result.fetchall(), columns=result.keys())

    # CSV 파일로 저장
    data.to_csv('data/user_movie_data.csv', index=False)
    print("CSV 파일 'data/user_movie_data.csv' 생성 완료!")
