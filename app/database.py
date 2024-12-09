import json
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# JSON 파일에서 DB 설정 로드
def load_db_config():
    with open("config/db_config.json", "r") as file:
        config = json.load(file)
    return config["database"]

# 데이터베이스 URL 생성
db_config = load_db_config()
DATABASE_URL = f"mysql+pymysql://{db_config['username']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['db_name']}"

# SQLAlchemy 세션 설정
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
