from sqlalchemy import text
import tensorflow as tf
import pickle
import numpy as np
from app.database import  fetch_all_users, fetch_movie_details, fetch_user_liked_movies, save_user_recommendations

# ========== RMSE 정의 ==========
def RMSE(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

# ========== 환경 설정 ==========
MODEL_PATH = "model/recommender_model/saved_model.keras"
METADATA_PATH = "model/recommender_model/metadata.pkl"

# ========== 모델 및 메타데이터 로드 ==========
try:
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects={"RMSE": RMSE})
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
except Exception as e:
    raise ValueError(f"모델 또는 메타데이터 로드 중 오류 발생: {e}")

tmdb2idx = metadata.get("tmdb2idx")
if not tmdb2idx:
    raise ValueError("'tmdb2idx' 키를 메타데이터에서 찾을 수 없습니다.")
idx_to_tmdb = {v: k for k, v in tmdb2idx.items()}

try:
    movie_embedding_layer = model.get_layer(name="embedding_1")
    movie_embedding_weights = movie_embedding_layer.get_weights()[0]
except Exception as e:
    raise ValueError(f"영화 임베딩 레이어 또는 가중치 로드 중 오류 발생: {e}")

# ========== 추천 시스템 함수 ==========
def get_user_vector(user_ratings, movie_embeddings, tmdb2idx):
    """사용자 평점을 기반으로 사용자 벡터 생성."""
    if not user_ratings:
        return np.zeros(movie_embeddings.shape[1], dtype=np.float32)

    user_vector = np.zeros(movie_embeddings.shape[1], dtype=np.float32)
    total_weight = 0.0

    for tmdb_id, rating in user_ratings.items():
        if tmdb_id in tmdb2idx:
            user_vector += movie_embeddings[tmdb2idx[tmdb_id]] * rating
            total_weight += rating

    return user_vector / total_weight if total_weight > 0 else user_vector

def calculate_similarities(user_vector, movie_embeddings):
    """사용자 벡터와 영화 임베딩 간 코사인 유사도 계산."""
    user_norm = np.linalg.norm(user_vector)

    if user_norm == 0:
        raise ValueError("유효하지 않은 사용자 벡터입니다. 벡터의 길이가 0입니다.")

    movie_norms = np.linalg.norm(movie_embeddings, axis=1)

    valid_mask = movie_norms != 0
    if not valid_mask.any():
        raise ValueError("유효한 영화 임베딩이 없습니다. 벡터의 길이가 0입니다.")

    similarities = np.zeros(movie_embeddings.shape[0])
    similarities[valid_mask] = (
        np.dot(movie_embeddings[valid_mask], user_vector) /
        (movie_norms[valid_mask] * user_norm)
    )
    return similarities


def recommend_movies(user_vector, movie_embeddings, idx_to_tmdb, user_ratings=None, top_k=10):
    """사용자 벡터 기반 추천."""
    similarities = calculate_similarities(user_vector, movie_embeddings)
    top_indices = np.argsort(-similarities)

    recommended = []
    seen_movies = set(user_ratings.keys()) if user_ratings else set()

    for idx in top_indices:
        if idx in idx_to_tmdb:
            tmdb_id = idx_to_tmdb[idx]
            if tmdb_id not in seen_movies:
                seen_movies.add(tmdb_id)
                recommended.append({"movie_id": int(tmdb_id), "similarity": float(similarities[idx])})
                if len(recommended) == top_k:
                    break
    return recommended


def recommend_similar_movies(liked_movies, movie_embeddings, idx_to_tmdb, top_k=10):
    """
    좋아요한 영화와 비슷한 영화를 추천.
    """
        # user_liked_movies가 무엇인지 출력해보기
    print(f"user_liked_movies: {liked_movies}")

    # 각 liked_movie_id에 대해 tmdb2idx에 키가 존재하는지 확인


    for liked_movie_id in liked_movies:
        if liked_movie_id in tmdb2idx:
           print(f"liked_movie_id {liked_movie_id}는 tmdb2idx에 존재합니다.")
        else:
            print(f"liked_movie_id {liked_movie_id}는 tmdb2idx에 없습니다.")
            
    liked_indices = [tmdb2idx[tmdb_id] for tmdb_id in liked_movies if tmdb_id in tmdb2idx]
    if not liked_indices:
        return []

    liked_embeddings = movie_embeddings[liked_indices]
    liked_vector = np.mean(liked_embeddings, axis=0)
    print("liked_embeddings:\n", liked_embeddings)
    similarities = calculate_similarities(liked_vector, movie_embeddings)
    top_indices = np.argsort(-similarities)

    recommended_movies = []
    for idx in top_indices:
        if idx in idx_to_tmdb:
            tmdb_id = idx_to_tmdb[idx]
            if tmdb_id not in liked_movies:
                recommended_movies.append({
                    "movie_id": int(tmdb_id),
                    "similarity": float(similarities[idx])
                })
                if len(recommended_movies) == top_k:
                    break

    return recommended_movies

def recommend_movies_combined(user_vector, user_liked_movies, movie_embeddings, idx_to_tmdb, db, user_ratings=None, top_k=10):

    user_recommendations = recommend_movies(user_vector, movie_embeddings, idx_to_tmdb, user_ratings, top_k)
    liked_recommendations = recommend_similar_movies(user_liked_movies, movie_embeddings, idx_to_tmdb, top_k)


    all_movie_ids = {rec["movie_id"] for rec in user_recommendations + liked_recommendations}
    movie_details = fetch_movie_details(list(all_movie_ids), db)

    movie_popularity = {movie["id"]: movie.get("popularity", 0) for movie in movie_details}


    seen_movies = set()
    combined_recommendations = []

    for rec in user_recommendations + liked_recommendations:
        movie_id = rec["movie_id"]
        if movie_id not in seen_movies:
            seen_movies.add(movie_id)
            rec["popularity"] = movie_popularity.get(movie_id, 0) 
            combined_recommendations.append(rec)


    for rec in combined_recommendations:
        rec["priority_score"] = rec["similarity"] * 0.4 + rec["popularity"] * 0.6
    combined_recommendations.sort(key=lambda x: x["priority_score"], reverse=True)

    return combined_recommendations[:top_k]


def generate_user_recommendation(user, movie_embeddings, tmdb2idx, idx_to_tmdb, db, top_k=10):
    user_vector = get_user_vector(user["ratings"], movie_embeddings, tmdb2idx)
    user_liked_movies = fetch_user_liked_movies(user["user_id"], db)

    if np.linalg.norm(user_vector) == 0:
        recommended_movies = recommend_similar_movies(user_liked_movies, movie_embeddings, idx_to_tmdb, top_k)
        return {"user_id": user["user_id"], "recommended_movies": recommended_movies}

    recommended_movies = recommend_movies_combined(
        user_vector, user_liked_movies, movie_embeddings, idx_to_tmdb, db ,user["ratings"], top_k
    )
    return {"user_id": user["user_id"], "recommended_movies": recommended_movies}

def generate_and_store_recommendations(movie_embeddings, tmdb2idx, idx_to_tmdb, db, top_k=10):
    try:

        db.execute(text("DELETE FROM recommend"))
        db.commit()
        print("기존 추천 데이터를 모두 삭제했습니다.")
    except Exception as e:
        db.rollback()
        print(f"Error deleting existing recommendations: {e}")
        return

    try:
        users = fetch_all_users(db)
    except Exception as e:
        print(f"Error fetching users: {e}")
        return

    all_recommendations = []
    for user in users:
        try:
            recommendation = generate_user_recommendation(user, movie_embeddings, tmdb2idx, idx_to_tmdb, db, top_k)
            all_recommendations.append(recommendation)
        except Exception as e:
            print(f"Error generating recommendation for user {user['user_id']}: {e}")

    try:
        save_user_recommendations(all_recommendations, db)
        print(f"{len(users)}명의 사용자 추천 결과를 저장했습니다.")
    except Exception as e:
        print(f"Error saving recommendations: {e}")

    for rec in all_recommendations:
        print(f"User {rec['user_id']} recommendations: {rec['recommended_movies']}")
