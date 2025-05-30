import os
from datetime import datetime
from unittest.mock import MagicMock  # For mocking API calls

import pytest
import requests
from news_recommender.data_ingestion.fetch_articles import (
	fetch_single_query_from_newsapi,
	get_num_articles,
	load_api_response_from_cache,
	save_api_response_to_cache,
)

# --- Tests for get_num_articles ---


def test_get_num_articles_sum_and_length():
	num_categories = 5
	target_num = 50
	result = get_num_articles(num_categories, target_num)
	assert len(result) == num_categories
	assert sum(result) == target_num


def test_get_num_articles_non_negative():
	result = get_num_articles(num_categories=3, target_num=10)
	for count in result:
		assert count >= 0


def test_get_num_articles_single_category():
	result = get_num_articles(num_categories=1, target_num=20)
	assert len(result) == 1
	assert result[0] == 20
	assert sum(result) == 20


# --- Tests for Caching Functions (using pytest's tmp_path fixture) ---


def test_save_and_load_cache(tmp_path):
	cache_dir = tmp_path / "api_cache"
	cache_dir.mkdir()
	cache_filename = cache_dir / "test_cache.json"
	sample_data = {"status": "ok", "articles": [{"title": "Test Article"}]}

	save_api_response_to_cache(str(cache_filename), sample_data)
	assert os.path.exists(cache_filename)

	loaded_data = load_api_response_from_cache(str(cache_filename))
	assert loaded_data == sample_data


def test_load_cache_non_existent(tmp_path):
	cache_filename = tmp_path / "api_cache" / "non_existent.json"
	loaded_data = load_api_response_from_cache(str(cache_filename))
	assert loaded_data is None


# --- Tests for fetch_single_query_from_newsapi (using mocking) ---

# Define a sample successful API response
SAMPLE_API_OK_RESPONSE = {
	"status": "ok",
	"totalResults": 1,
	"articles": [
		{
			"source": {"id": None, "name": "Test Source"},
			"title": "Test Article 1",
			"url": "http://example.com/article1",
		}
	],
}

# Define a sample API error response
SAMPLE_API_ERROR_RESPONSE = {
	"status": "error",
	"code": "apiKeyInvalid",
	"message": "Your API key is invalid or incorrect.",
}


@pytest.fixture
def mock_env_vars(monkeypatch):
	"""Fixture to mock env variables like API_KEY and set CACHE_DIR for tests."""
	monkeypatch.setenv("NEWSAPI_KEY", "test_api_key_for_pytest")


def test_fetch_single_query_cache_miss_then_hit(mocker, tmp_path, mock_env_vars):
	category = "technology"
	page_size = 10

	# Make CACHE_DIR a global variable available to the SUT (System Under Test) funcs
	# This is a bit tricky as the SUT uses a global CACHE_DIR.
	# For tests, it's better if CACHE_DIR is configurable or passed to functions.
	# Here, we'll patch the module-level CACHE_DIR used by the functions.
	# Assuming your script is `src.news_recommender.data_ingestion.fetch_articles`
	mocker.patch(
		"news_recommender.data_ingestion.fetch_articles.CACHE_DIR", str(tmp_path)
	)
	# And ensure API_KEY is available if it's accessed globally in the SUT
	mocker.patch(
		"news_recommender.data_ingestion.fetch_articles.API_KEY",
		"test_api_key_for_pytest",
	)

	# --- First Call (Cache Miss, API Call) ---
	mock_response = MagicMock()
	mock_response.status_code = 200
	mock_response.json.return_value = SAMPLE_API_OK_RESPONSE
	mock_requests_get = mocker.patch("requests.get", return_value=mock_response)

	data1 = fetch_single_query_from_newsapi(category, page_size)

	mock_requests_get.assert_called_once()  # Check API was called
	assert data1 == SAMPLE_API_OK_RESPONSE

	# Check if cache file was created
	today_date_str = datetime.now().strftime(
		"%Y-%m-%d"
	)  # Use now() for test consistency if test runs fast
	# Reconstruct expected cache filename (ensure it matches your function's logic)
	safe_category_name = "".join(c if c.isalnum() else "_" for c in category)
	expected_cache_file_base = f"{str(today_date_str)}_{safe_category_name}_{page_size}"
	expected_cache_filename = tmp_path / f"{expected_cache_file_base}.json"
	assert os.path.exists(expected_cache_filename)

	# --- Second Call (Cache Hit) ---
	# Reset mock to see if it's called again (it shouldn't be)
	mock_requests_get.reset_mock()

	data2 = fetch_single_query_from_newsapi(category, page_size)

	mock_requests_get.assert_not_called()  # API should NOT be called
	assert data2 == SAMPLE_API_OK_RESPONSE  # Data should come from cache


def test_fetch_single_query_api_status_error(mocker, tmp_path, mock_env_vars):
	mocker.patch(
		"news_recommender.data_ingestion.fetch_articles.CACHE_DIR", str(tmp_path)
	)
	mocker.patch(
		"news_recommender.data_ingestion.fetch_articles.API_KEY",
		"test_api_key_for_pytest",
	)

	mock_response = MagicMock()
	mock_response.status_code = 200  # API call itself was successful
	mock_response.json.return_value = (
		SAMPLE_API_ERROR_RESPONSE  # But API reports an error in its JSON
	)
	mock_requests_get = mocker.patch("requests.get", return_value=mock_response)

	# Ensure cache is empty for this specific test to force API call
	data = fetch_single_query_from_newsapi("business_error", 5)

	mock_requests_get.assert_called_once()
	assert data is None  # Function should return None on API error status


def test_fetch_single_query_http_error(mocker, tmp_path, mock_env_vars):
	mocker.patch(
		"news_recommender.data_ingestion.fetch_articles.CACHE_DIR", str(tmp_path)
	)
	mocker.patch(
		"news_recommender.data_ingestion.fetch_articles.API_KEY",
		"test_api_key_for_pytest",
	)

	mock_response = MagicMock()
	mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
		"Test HTTP Error"
	)
	mock_requests_get = mocker.patch("requests.get", return_value=mock_response)

	data = fetch_single_query_from_newsapi("health_http_error", 7)

	mock_requests_get.assert_called_once()
	mock_response.raise_for_status.assert_called_once()
	assert data is None  # Function should return None on HTTP error


def test_fetch_single_query_request_exception(mocker, tmp_path, mock_env_vars):
	mocker.patch(
		"news_recommender.data_ingestion.fetch_articles.CACHE_DIR", str(tmp_path)
	)
	mocker.patch(
		"news_recommender.data_ingestion.fetch_articles.API_KEY",
		"test_api_key_for_pytest",
	)

	mock_requests_get = mocker.patch(
		"requests.get",
		side_effect=requests.exceptions.ConnectionError("Test Connection Error"),
	)

	data = fetch_single_query_from_newsapi("science_conn_error", 8)

	mock_requests_get.assert_called_once()
	assert data is None  # Function should return None on RequestException
