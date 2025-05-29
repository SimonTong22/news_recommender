import os

from dotenv import load_dotenv

load_dotenv()  # Loads variables from .env into environment variables
API_KEY = os.getenv("NEWSAPI_KEY")

if not API_KEY:
	raise ValueError(
		"NEWSAPI_KEY not found in environment variables. Did you create a .env file?"
	)
