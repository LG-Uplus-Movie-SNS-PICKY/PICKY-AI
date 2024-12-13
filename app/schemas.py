from pydantic import BaseModel
from typing import List

class UserRating(BaseModel):
    moive_id: int
    rating: float

class UserRatings(BaseModel):
    user_id: int
    ratings: List[UserRating]
