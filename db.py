# db.py
# This file handles all database operations for MongoDB.
# DEFINITIVE FIX: Replaced bcrypt with Argon2 to permanently remove password length limits.

import streamlit as st
from pymongo import MongoClient
from passlib.context import CryptContext
from datetime import datetime
import certifi

# --- DATABASE CONNECTION ---
try:
    client = MongoClient(st.secrets["MONGO_URI"], tlsCAFile=certifi.where())
    db = client.crimelens_db
    users_collection = db.users
    activity_log_collection = db.activity_logs
    st.success("Connected to MongoDB!")
except Exception as e:
    st.error(f"Error connecting to MongoDB: {e}")

# --- PASSWORD HASHING ---
# ✨ --- THE FIX: Using Argon2 instead of Bcrypt --- ✨
# Argon2 is modern, more secure, and has no password length limitations.
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

def verify_password(plain_password, hashed_password):
    """Verifies a plain password against a hashed one using Argon2."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    """Hashes a password for storing using Argon2."""
    return pwd_context.hash(password)

# --- USER AUTHENTICATION FUNCTIONS ---
# These functions are now simpler and more robust.
def sign_up(username, password):
    """Signs up a new user. Returns True on success, False otherwise."""
    if not username or not password:
        return False
    if users_collection.find_one({"username": username}):
        return False
    try:
        hashed_password = get_password_hash(password)
        users_collection.insert_one({"username": username, "password": hashed_password})
        return True
    except Exception as e:
        # Catch any other potential errors during DB operation
        print(f"Error during sign-up: {e}")
        return False

def login(username, password):
    """Logs in a user. Returns True on success, False otherwise."""
    if not username or not password:
        return False
        
    user = users_collection.find_one({"username": username})
    # The verify function will handle everything, including checking if the hash format is correct.
    if user and verify_password(password, user["password"]):
        return True
    return False

# --- ACTIVITY LOGGING FUNCTION ---
def log_activity(username, activity_type, details):
    """Logs a user's activity to the database."""
    log_entry = {
        "username": username,
        "activity_type": activity_type,
        "timestamp": datetime.now(),
        "details": details
    }
    activity_log_collection.insert_one(log_entry)

