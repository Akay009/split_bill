import os
from dotenv import load_dotenv
load_dotenv()

mongo_url = os.getenv("MONGO_URL")