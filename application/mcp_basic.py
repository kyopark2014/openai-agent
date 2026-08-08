import logging
import sys
import datetime
import requests
import traceback
import json
import re
import utils
import os

from pytz import timezone
from bs4 import BeautifulSoup

import bedrock_responses

logging.basicConfig(
    level=logging.INFO,
    format="%(filename)s:%(lineno)d | %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger("mcp-basic")

MODEL_ID = "openai.gpt-5.5"
REGION = "us-east-2"


_KOREAN_WEEKDAYS = ("월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일")

def get_current_time(format: str="%Y-%m-%d %H:%M:%S")->str:
    """Returns the current date and time in Asia/Seoul, including the Korean weekday.

    Example: "2026-08-08 15:51:06 (토요일)"
    """
    format = format.replace("'", "")
    now = datetime.datetime.now(timezone('Asia/Seoul'))
    timestr = f"{now.strftime(format)} ({_KOREAN_WEEKDAYS[now.weekday()]})"
    logger.info(f"timestr: {timestr}")
    
    return timestr


def get_book_list(keyword: str) -> str:
    """Search book list by keyword and return book list."""
    keyword = keyword.replace("'", "")
    answer = ""
    url = f"https://search.kyobobook.co.kr/search?keyword={keyword}&gbCode=TOT&target=total"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        prod_info = soup.find_all("a", attrs={"class": "prod_info"})
        if len(prod_info):
            answer = "추천 도서는 아래와 같습니다.\n"
        for prod in prod_info[:5]:
            title = prod.text.strip().replace("\n", "")
            link = prod.get("href")
            answer = answer + f"{title}, URL: {link}\n\n"
    return answer


def isKorean(text):
    pattern_hangul = re.compile("[\u3131-\u3163\uac00-\ud7a3]+")
    word_kor = pattern_hangul.search(str(text))
    return bool(word_kor and word_kor != "None")


def traslation(text, input_language, output_language):
    instructions = (
        f"Translate {input_language} to {output_language}. "
        "Put the translation in <result> tags."
    )
    user_input = f"<article>{text}</article>"
    try:
        msg = bedrock_responses.complete_text(
            instructions=instructions,
            user_input=user_input,
            model_id=MODEL_ID,
            region=REGION,
        )
    except Exception:
        err_msg = traceback.format_exc()
        logger.info(f"error message: {err_msg}")
        raise Exception("Not able to request to LLM")

    if "<result>" in msg:
        return msg[msg.find("<result>") + 8 : msg.find("</result>")]
    return msg


def get_weather_info(city: str) -> str:
    """Retrieve weather information by city name."""
    city = city.replace("\n", "").replace("'", "").replace('"', "")

    if isKorean(city):
        place = traslation(city, "Korean", "English")
        logger.info(f"city (translated): {place}")
    else:
        place = city
        city = traslation(city, "English", "Korean")
        logger.info(f"city (translated): {city}")

    logger.info(f"place: {place}")
    weather_str: str = f"{city}에 대한 날씨 정보가 없습니다."

    weather_api_key = utils.weather_api_key
    if weather_api_key:
        api = (
            f"https://api.openweathermap.org/data/2.5/weather?q={place}"
            f"&APPID={weather_api_key}&lang=en&units=metric"
        )
        try:
            result = json.loads(requests.get(api).text)
            logger.info(f"result: {result}")
            if "weather" in result:
                overall = result["weather"][0]["main"]
                current_temp = result["main"]["temp"]
                humidity = result["main"]["humidity"]
                wind_speed = result["wind"]["speed"]
                cloud = result["clouds"]["all"]
                weather_str = (
                    f"{city}의 현재 날씨의 특징은 {overall}이며, 현재 온도는 {current_temp} 입니다. "
                    f"현재 습도는 {humidity}% 이고, 바람은 초당 {wind_speed} 미터 입니다. "
                    f"구름은 {cloud}% 입니다."
                )
        except Exception:
            err_msg = traceback.format_exc()
            logger.info(f"error message: {err_msg}")

    logger.info(f"weather_str: {weather_str}")
    return weather_str
