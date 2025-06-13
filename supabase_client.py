from typing import List, Optional, Union
from datetime import datetime
from pydantic import BaseModel
from supabase import create_client, Client
from dotenv import load_dotenv
import os
# Load environment variables
load_dotenv('config/.env')

# Initialize Supabase client
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")

print(url)
print(key)

supabase: Client = create_client(url, key)

class Experience(BaseModel):
    title: str
    time: str
    transportation: str
    standard_rate: Union[str, List[str]]
    standard_rate_description: str
    booking_summary: str
    description: str
    start_time: List[str]
    experience_type: str
    attributes: str
    categories: str
    booking_in_advance: str
    live_tour_guide: str

# Clase Pydantic para embedding_experiences
class EmbeddingExperience(BaseModel):
    id: Optional[int] = None
    experience_id: int
    title: str
    text_church: str
    embedding: List[float]

class Booking(BaseModel):
    id: int
    pick_up_date: Optional[str] = None
    pick_up_time: Optional[str] = None
    number_of_passengers: Optional[int] = None
    luggage_count: Optional[int] = None
    number_adults: Optional[int] = None
    number_children: Optional[int] = None
    number_infant: Optional[int] = None
    amount: Optional[float] = None
    reservation_status: Optional[str] = None
    cancellation_policy: Optional[str] = None
    refund_protection: Optional[float] = None
    language: Optional[str] = None
    payment_type: Optional[str] = None
    status: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    room: Optional[str] = None
    ticket: Optional[str] = None
    code: Optional[str] = None
    payment_status: Optional[bool] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    published_at: Optional[str] = None
    created_by_id: Optional[int] = None
    updated_by_id: Optional[int] = None
    deposit_amount: Optional[float] = None
    paid: Optional[float] = None
    tour_id: Optional[int] = None
    reference_number: Optional[str] = None

def create_experience(experience: Experience) -> dict:
    """Create a new experience in the database"""
    data = experience.model_dump()
    # Insert into experiences table
    result = supabase.table('experiences').insert(data).execute()
    return result.data

def get_experience(id: int) -> Optional[dict]:
    """Get an experience by ID"""
    result = supabase.table('experiences').select("*").eq('id', id).execute()
    return result.data[0] if result.data else None

def update_experience(id: int, experience: Experience) -> dict:
    """Update an existing experience"""
    data = experience.model_dump()
    result = supabase.table('experiences').update(data).eq('id', id).execute()
    return result.data

def delete_experience(id: int) -> dict:
    """Delete an experience"""
    result = supabase.table('experiences').delete().eq('id', id).execute()
    return result.data

def search_similar_experiences(query: str, limit: int = 5) -> List[dict]:
    """Search for similar experiences using pgvector"""
    # Using pgvector's cosine similarity
    result = supabase.rpc(
        'match_experiences',
        {
            'query_embedding': query,
            'match_threshold': 0.7,
            'match_count': limit
        }
    ).execute()
    return result.data

def search_similar_embedding_experiences(query: list, limit: int = 5) -> List[dict]:
    """Search for similar experiences using pgvector"""
    # Using pgvector's cosine similarity
    result = supabase.rpc(
        'match_embedding_experiences_evaluation',
        {
            'query_embedding': query,
            'match_threshold': 0.6,
            'match_count': limit
        }
    ).execute()
    return result

def get_all_experiences() -> List[dict]:
    """
    Get all experiences from the database.
    
    Returns:
        List[dict]: A list of all experiences in the database
    """
    result = supabase.table('experiences').select("*").execute()
    return result.data

def create_embedding_experience(embedding_experience: EmbeddingExperience) -> dict:
    """Create a new embedding experience in the database"""
    data = embedding_experience.model_dump(exclude_unset=True)
    result = supabase.table('embedding_experiences').insert(data).execute()
    return result.data

def get_embedding_experience(id: int) -> Optional[dict]:
    """Get an embedding experience by ID"""
    result = supabase.table('embedding_experiences').select("*").eq('id', id).execute()
    return result.data[0] if result.data else None

def update_embedding_experience(id: int, embedding_experience: EmbeddingExperience) -> dict:
    """Update an existing embedding experience"""
    data = embedding_experience.model_dump(exclude_unset=True)
    result = supabase.table('embedding_experiences').update(data).eq('id', id).execute()
    return result.data

def delete_embedding_experience(id: int) -> dict:
    """Delete an embedding experience"""
    result = supabase.table('embedding_experiences').delete().eq('id', id).execute()
    return result.data

def get_all_embedding_experiences() -> List[dict]:
    """Get all embedding experiences from the database"""
    result = supabase.table('embedding_experiences').select("*").execute()
    return result.data

def create_booking(booking: Booking) -> dict:
    """Create a new booking in the database"""
    data = booking.model_dump()
    result = supabase.table('bookings').insert(data).execute()
    return result.data

def get_booking(id: int) -> Optional[dict]:
    """Get a booking by ID"""
    result = supabase.table('bookings').select("*").eq('id', id).execute()
    return result.data[0] if result.data else None

def update_booking(id: int, booking: Booking) -> dict:
    """Update an existing booking"""
    data = booking.model_dump()
    result = supabase.table('bookings').update(data).eq('id', id).execute()
    return result.data

def delete_booking(id: int) -> dict:
    """Delete a booking"""
    result = supabase.table('bookings').delete().eq('id', id).execute()
    return result.data

def list_bookings(limit: int = 100) -> list:
    """List bookings (default limit 100)"""
    result = supabase.table('bookings').select("*").limit(limit).execute()
    return result.data
