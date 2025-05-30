import hashlib
import json
import os
from datetime import datetime

import numpy as np
import requests
from dotenv import load_dotenv
from scipy.stats import truncnorm

NEWSAPI_BASE_URL = "https://newsapi.org/v2/top-headlines"
RAW_ARTICLES_DIR = "data/raw/articles"
CACHE_DIR = "data/raw/api_cache"
os.makedirs(RAW_ARTICLES_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

load_dotenv()
API_KEY = os.getenv("NEWSAPI_KEY")

if not API_KEY:
	raise ValueError(
		"NEWSAPI_KEY not found in environment variables. Did you create a .env file?"
	)

NEWS_CATEGORIES = [
	"business",
	"entertainment",
	"general",
	"health",
	"science",
	"sports",
	"technology",
]
API_LIMIT = 100


def get_num_articles(num_categories, target_num):
	# Get a random number of articles per category
	# Sampling from the normal distribution
	# The total articles should respect the API_LIMIT
	# (or target_num)
	articles_per_cat = target_num / num_categories
	std_dev = articles_per_cat / 2
	lower_bound = 0
	upper_bound = target_num
	a = (lower_bound - articles_per_cat) / std_dev
	b = (upper_bound - articles_per_cat) / std_dev

	samples = truncnorm.rvs(
		a, b, loc=articles_per_cat, scale=std_dev, size=num_categories
	)

	rounded_samples = [max(0, round(sample)) for sample in samples]
	current_sum = sum(rounded_samples)
	difference = target_num - current_sum
	while difference != 0:
		if difference > 0:  # Need to add articles
			# Add to a random category or the one with the fewest articles
			idx_to_change = np.random.choice(num_categories)
			rounded_samples[idx_to_change] += 1
			difference -= 1
		else:  # Need to subtract articles (difference < 0)
			# Subtract from a random category that has more than 0
			eligible_indices = [
				i for i, count in enumerate(rounded_samples) if count > 0
			]
			idx_to_change = np.random.choice(eligible_indices)
			rounded_samples[idx_to_change] -= 1
			difference += 1

	return rounded_samples


def save_api_response_to_cache(cache_filename, data):
	"""Saves API response to a cache file."""
	print(f"Saving to cache: {cache_filename}")
	with open(cache_filename, "w", encoding="utf-8") as f:
		json.dump(data, f, ensure_ascii=False, indent=4)


def load_api_response_from_cache(cache_filename):
	"""Loads API response from a cache file."""
	if os.path.exists(cache_filename):
		print(f"Loading from cache: {cache_filename}")
		with open(cache_filename, encoding="utf-8") as f:
			response = json.load(f)
			return response
	return None


def fetch_single_query_from_newsapi(
	category,
	page_size,
):
	today_date_str = datetime.today().date()
	cache_filename = f"{today_date_str}_{category}_{page_size}"
	full_cache_filename = os.path.join(CACHE_DIR, f"{cache_filename}.json")
	cached_data = load_api_response_from_cache(full_cache_filename)
	if cached_data:
		return cached_data
	header = {"X-Api-Key": API_KEY}
	params = {"category": category, "pageSize": page_size}
	try:
		response = requests.get(NEWSAPI_BASE_URL, headers=header, params=params)
		response.raise_for_status()
		data = response.json()
		if data.get("status") == "ok":
			save_api_response_to_cache(full_cache_filename, data)
			return data
		else:
			print(f"API Error: {data.get('message')}")
			return None
	except requests.exceptions.RequestException as e:
		print(f"Request failed: {e}")
		return None


def fetch_all_articles(categories, articles_per_category):
	today_date_str = datetime.today().date()
	today_output_dir = os.path.join(RAW_ARTICLES_DIR, today_date_str)
	os.makedirs(today_output_dir, exist_ok=True)

	all_fetched_articles = []
	article_urls = set()
	for category, num_articles in zip(categories, articles_per_category, strict=False):
		data = fetch_single_query_from_newsapi(category, num_articles)
		if data and data.get("articles"):
			for article in data["articles"]:
				article_url = article.get("url")
				if article_url not in article_urls:
					all_fetched_articles.append(article)
					article_urls.add(article.get("url"))
					# For article file naming, we want to ensure uniqueness
					# between different articles
					# But we also want to make sure that filenames are consistent
					# for the same article
					# Even if they are found on different days/queries
					if article_url:
						# Create a filename based on a hash of the URL
						hashed_url = hashlib.md5(
							article_url.encode("utf-8")
						).hexdigest()
						article_filename = f"{hashed_url}"
					else:
						# Use the title if there is no url
						article_filename = article.get(
							"title", f"untitled_{len(article_urls)}"
						)[:50]
					article_filepath = os.path.join(
						today_output_dir,
						f"{article_filename}.json",
					)
					with open(article_filepath, "w", encoding="utf-8") as f:
						json.dump(article, f, ensure_ascii=False, indent=4)

	print(
		"Finished fetching. Total unique articles collected: "
		f"{len(all_fetched_articles)}"
	)


if __name__ == "__main__":
	articles_per_category = get_num_articles(
		num_categories=len(NEWS_CATEGORIES), target_num=API_LIMIT
	)
	fetch_all_articles(NEWS_CATEGORIES, articles_per_category)
