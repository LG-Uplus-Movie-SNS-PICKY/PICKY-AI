import tensorflow as tf
import pickle
import numpy as np
import logging
from app.database import fetch_movie_details, fetch_all_users, save_user_recommendations

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    if not user_ratings:
        # 신규 사용자 또는 평가 데이터가 없는 사용자 처리
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
    else:
        # 평가 데이터는 있지만 매핑이 안 된 경우
        user_vector = np.zeros(embedding_dim, dtype=np.float32)

    return user_vector


def calculate_similarities(user_vector, movie_embeddings):
    user_norm = np.linalg.norm(user_vector)
    if user_norm == 0:
        raise ValueError("사용자 벡터의 길이가 0입니다. 벡터가 올바르게 생성되었는지 확인하세요.")

    movie_norms = np.linalg.norm(movie_embeddings, axis=1)
    non_zero_mask = movie_norms != 0

    similarities = np.zeros(movie_embeddings.shape[0])
    valid_movie_embeddings = movie_embeddings[non_zero_mask]
    similarities[non_zero_mask] = np.dot(valid_movie_embeddings, user_vector) / (
        movie_norms[non_zero_mask] * user_norm
    )

    return similarities

def recommend_movies(user_vector, movie_embeddings, idx_to_tmdb, user_ratings=None, top_k=10):
    similarities = calculate_similarities(user_vector, movie_embeddings)
    top_indices = np.argsort(-similarities)

    recommended_movies = []
    for idx in top_indices:
        if idx in idx_to_tmdb:
            tmdb_id = idx_to_tmdb[idx]
            if user_ratings is None or tmdb_id not in user_ratings:
                recommended_movies.append({
                    "movie_id": int(tmdb_id),  # np.int64를 int로 변환
                    "similarity": float(similarities[idx])  # similarity 값 포함
                })
                if len(recommended_movies) == top_k:
                    break

    if not recommended_movies:
        print("No recommendations generated.")

    return recommended_movies



def generate_user_recommendation(user, movie_embeddings, tmdb_to_idx, idx_to_tmdb, top_k=10):
    user_vector = get_user_vector(user["ratings"], movie_embeddings, tmdb_to_idx)
    recommended_tmdb_ids = recommend_movies(user_vector, movie_embeddings, idx_to_tmdb, user["ratings"], top_k)
    return {"user_id": user["user_id"], "recommended_movies": recommended_tmdb_ids}

def generate_and_store_recommendations(movie_embeddings, tmdb_to_idx, idx_to_tmdb, db, top_k=10):
    try:
        users = fetch_all_users(db)
    except Exception as e:
        print(f"Error fetching users: {e}")
        return

    all_recommendations = []
    for user in users:
        try:
            recommendation = generate_user_recommendation(user, movie_embeddings, tmdb_to_idx, idx_to_tmdb, top_k)
            all_recommendations.append(recommendation)
        except Exception as e:
            print(f"Error generating recommendation for user {user['user_id']}: {e}")

    try:
        save_user_recommendations(all_recommendations, db)
        print(f"{len(users)}명의 사용자 추천 결과를 저장했습니다.")
    except Exception as e:
        print(f"Error saving recommendations: {e}")

    # 추천 결과를 출력
    for rec in all_recommendations:
        print(f"User {rec['user_id']} recommendations: {rec['recommended_movies']}")

