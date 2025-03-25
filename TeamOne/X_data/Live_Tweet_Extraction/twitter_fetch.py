import requests
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import psycopg2

print("🚀 All comapnies 50 tweets Final")

# Twitter API config
API_KEY = "93af433a8ff843819d702acbfadcc895"
base_url = "https://api.twitterapi.io/twitter/tweet/advanced_search"
headers = {"X-API-Key": API_KEY}