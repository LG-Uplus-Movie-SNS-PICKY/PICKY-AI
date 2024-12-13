import tensorflow as tf
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from app.database import fetch_movie_details

def RMSE(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

# ========== 환경 설정 (파일 경로나 DB 연결 정보 등) ==========
MODEL_PATH = "model/recommender_model/saved_model.keras"
METADATA_PATH = "model/recommender_model/metadata.pkl"

# ========== 모델 및 메타데이터 로드 ==========
try:
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects={"RMSE": RMSE})
except Exception as e:
    raise ValueError(f"모델 로드 중 오류 발생: {e}")

try:
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
except FileNotFoundError:
    raise FileNotFoundError("메타데이터 파일을 찾을 수 없습니다. 경로를 확인하세요.")
except Exception as e:
    raise ValueError(f"메타데이터 로드 중 오류 발생: {e}")

tmdb_to_idx = metadata.get("tmdb_to_idx")
if tmdb_to_idx is None:
    raise ValueError("메타데이터에서 'tmdb_to_idx' 키를 찾을 수 없습니다.")

idx_to_tmdb = {v: k for k, v in tmdb_to_idx.items()}

# 영화 임베딩 레이어 찾기
movie_embedding_layer = model.get_layer(name="embedding_1")
if movie_embedding_layer is None:
    raise ValueError("TMDb embedding layer를 찾지 못했습니다. 모델 구조를 확인하세요.")

movie_embedding_weights = movie_embedding_layer.get_weights()[0]
if movie_embedding_weights is None:
    raise ValueError("영화 임베딩 가중치를 로드할 수 없습니다.")

# ========== 함수 정의 ==========

def get_user_vector(user_ratings, movie_embeddings, tmdb_to_idx):
    """
    사용자 평점을 기반으로 사용자 벡터를 생성합니다.
    user_ratings: {tmdb_id: rating} 형태의 딕셔너리
    movie_embeddings: (num_movies+1, embedding_dim)
    tmdb_to_idx: {tmdb_id: embedding_index} 매핑 딕셔너리
    """
    if not user_ratings:
        # 평점이 없는 신규 사용자 처리 로직
        embedding_dim = movie_embeddings.shape[1]
        return np.zeros(embedding_dim, dtype=np.float32)

    embedding_dim = movie_embeddings.shape[1]
    user_vector = np.zeros(embedding_dim, dtype=np.float32)
    total_weight = 0.0

    for tmdb_id, rating in user_ratings.items():
        if tmdb_id in tmdb_to_idx:
            movie_idx = tmdb_to_idx[tmdb_id]
            user_vector += movie_embeddings[movie_idx] * rating
            total_weight += rating

    if total_weight > 0:
        user_vector /= total_weight
    return user_vector

def recommend_movies(user_vector, movie_embeddings, idx_to_tmdb, user_ratings=None, top_k=10):
    """
    사용자 벡터와 영화 임베딩을 기반으로 영화를 추천합니다.
    user_vector: 사용자 벡터
    movie_embeddings: 영화 임베딩 배열
    idx_to_tmdb: {embedding_index: tmdb_id} 매핑 딕셔너리
    user_ratiㄹgs: 사용자가 이미 평가한 영화 (선택사항)
    top_k: 추천할 영화 수
    """
    similarities = cosine_similarity([user_vector], movie_embeddings)[0]
    top_indices = np.argsort(-similarities)

    recommended_tmdb_ids = []
    for idx in top_indices:
        if idx in idx_to_tmdb:
            tmdb_id = idx_to_tmdb[idx]
            if user_ratings is None or tmdb_id not in user_ratings:
                recommended_tmdb_ids.append(tmdb_id)
                print(f"추천 ID 추가: {tmdb_id} (유사도: {similarities[idx]:.4f})")
                if len(recommended_tmdb_ids) == top_k:
                    break
    print(f"최종 추천 리스트: {recommended_tmdb_ids}")
    return recommended_tmdb_ids

def get_movie_details(tmdb_ids, db):
    tmdb_ids = [int(id) for id in tmdb_ids]  # Ensure all IDs are integers
    """
    TMDb ID를 기반으로 영화 세부 정보를 가져옵니다.
    tmdb_ids: TMDb ID 리스트
    db: 데이터베이스 연결 객체
    """
    if not tmdb_ids:
        return []
    try:
        movie_details = fetch_movie_details(tmdb_ids, db)
        if not movie_details:
            print(f"No details found for TMDb IDs: {tmdb_ids}")
        return movie_details
    except Exception as e:
        print(f"Error fetching movie details: {e}")
        return []
