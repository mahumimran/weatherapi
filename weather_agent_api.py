"""
Weather Agent API
==================
A FastAPI service that exposes a LangChain tool-calling agent capable of
answering natural-language weather queries. The agent's `get_current_weather`
tool calls the REAL OpenWeatherMap API for live conditions.

LLM BACKEND: Runs on Groq's cloud API — fast inference, generous free tier,
no local install/download required (unlike Ollama).

SETUP (one-time):
    1. Get a FREE Groq API key: https://console.groq.com/keys
    2. Get a FREE OpenWeatherMap API key: https://openweathermap.org/api
       (Sign up -> API keys tab -> copy the default key. New keys can take
       a few minutes to activate.)
    3. Put both in a .env file next to this script:
           GROQ_API_KEY=your_groq_key_here
           OPENWEATHER_API_KEY=your_openweather_key_here

Run with:
    python weather_agent_api.py

Environment variables (create a .env file or export directly):
    GROQ_API_KEY=...            # REQUIRED — LLM access
    OPENWEATHER_API_KEY=...     # REQUIRED — real weather data
    GROQ_MODEL=openai/gpt-oss-20b   # optional, default shown (tool-calling capable)
"""

import os
import logging
from typing import Optional

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from langchain_core.tools import tool
from langchain_groq import ChatGroq  # Groq cloud LLM — fast, free tier available
from langgraph.prebuilt import create_react_agent  # current stable agent-builder

import uvicorn

# --------------------------------------------------------------------------- #
# Environment & Logging Setup
# --------------------------------------------------------------------------- #

load_dotenv()  # loads variables from a .env file if present

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")  # required for real weather data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("weather_agent_api")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not set. Get a free key at https://console.groq.com/keys "
        "and add it to your environment or .env file."
    )

if not OPENWEATHER_API_KEY:
    # Not fatal at startup (so the app can still boot), but every weather
    # lookup will fail until this is set. Get a free key at openweathermap.org/api
    logger.warning(
        "OPENWEATHER_API_KEY is not set. Weather lookups will return an error "
        "until you add it to your environment or .env file."
    )


# --------------------------------------------------------------------------- #
# Pydantic Models (Request / Response Schemas)
# --------------------------------------------------------------------------- #

class WeatherQueryRequest(BaseModel):
    """Incoming request payload for the /api/weather endpoint."""
    user_query: str = Field(
        ...,
        min_length=1,
        description="Natural language weather question, e.g. 'What's the weather in Lahore?'",
        examples=["What's the weather like in Tokyo right now?"],
    )


class WeatherQueryResponse(BaseModel):
    """Outgoing response payload for the /api/weather endpoint."""
    user_query: str
    agent_response: str
    success: bool = True
    error: Optional[str] = None


# --------------------------------------------------------------------------- #
# LangChain Tool Definition
# --------------------------------------------------------------------------- #

@tool
def get_current_weather(location: str) -> str:
    """
    Get the current weather conditions for a given location.

    Args:
        location: The city or place name to fetch weather for (e.g. "Paris", "New York").

    Returns:
        A short string describing the temperature (Celsius) and weather conditions,
        or a clear error message if the location could not be found / the call failed.
    """
    # ------------------------------------------------------------------ #
    # REAL OPENWEATHERMAP IMPLEMENTATION
    # ------------------------------------------------------------------ #
    # Requires OPENWEATHER_API_KEY to be set (see .env / environment setup).
    # Get a free key at: https://openweathermap.org/api
    # ------------------------------------------------------------------ #
    if not OPENWEATHER_API_KEY:
        return (
            "Weather lookup is not configured: OPENWEATHER_API_KEY is missing. "
            "Please set it in your environment or .env file."
        )

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": location,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric",  # returns temperature directly in Celsius
    }

    try:
        resp = requests.get(url, params=params, timeout=6)

        if resp.status_code == 404:
            return f"I couldn't find weather data for '{location}'. Please check the spelling."

        resp.raise_for_status()
        data = resp.json()

        temp = data["main"]["temp"]
        feels_like = data["main"]["feels_like"]
        humidity = data["main"]["humidity"]
        condition = data["weather"][0]["description"]
        wind_speed = data["wind"]["speed"]
        city_name = data.get("name", location)

        return (
            f"Current weather in {city_name}: {temp}°C (feels like {feels_like}°C), "
            f"{condition}, humidity {humidity}%, wind speed {wind_speed} m/s."
        )

    except requests.exceptions.Timeout:
        return f"The weather service timed out while fetching data for {location}. Please try again."
    except requests.exceptions.RequestException as exc:
        logger.error(f"OpenWeatherMap request failed for '{location}': {exc}")
        return f"Sorry, I couldn't retrieve weather data for {location} right now."


TOOLS = [get_current_weather]


# --------------------------------------------------------------------------- #
# Agent Setup (LLM + Prompt + Tool-Calling Agent + Executor)
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = (
    "You are WeatherBot, a friendly and modern weather assistant. "
    "You help users understand current weather conditions in a clear, concise, "
    "and polite tone. Always use the get_current_weather tool when the user "
    "asks about weather for a specific location, rather than guessing. "
    "If the location is ambiguous or missing, politely ask the user to clarify. "
    "Keep your final answers short, friendly, and easy to read — plain text only."
)

llm = ChatGroq(
    model=GROQ_MODEL,      # e.g. "openai/gpt-oss-20b", "llama-3.3-70b-versatile" —
                            # any current tool-calling-capable model on Groq
    api_key=GROQ_API_KEY,
    temperature=0.3,
)

# create_react_agent builds a full tool-calling agent graph in one call —
# it handles the LLM call, the tool-execution loop, and routing internally.
agent_executor = create_react_agent(
    model=llm,
    tools=TOOLS,
    prompt=SYSTEM_PROMPT,
)


# --------------------------------------------------------------------------- #
# FastAPI Application
# --------------------------------------------------------------------------- #

app = FastAPI(
    title="Weather Agent API",
    description="A LangChain-powered agent that answers weather questions via FastAPI.",
    version="1.0.0",
)

# Allow the frontend (opened as a local HTML file or served from another port)
# to call this API from the browser. Tighten allow_origins in production —
# "*" is fine for a university/demo project.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Health"])
async def root():
    """Basic health check endpoint."""
    return {"status": "ok", "service": "weather-agent-api"}


@app.post("/api/weather", response_model=WeatherQueryResponse, tags=["Weather"])
async def query_weather(request: WeatherQueryRequest) -> WeatherQueryResponse:
    """
    Accepts a natural-language weather query, runs it through the LangChain
    tool-calling agent, and returns the agent's final text response.
    """
    try:
        result = await agent_executor.ainvoke(
            {"messages": [{"role": "user", "content": request.user_query}]}
        )
        final_output = result["messages"][-1].content.strip()

        if not final_output:
            raise ValueError("Agent returned an empty response.")

        return WeatherQueryResponse(
            user_query=request.user_query,
            agent_response=final_output,
            success=True,
        )

    except Exception as exc:
        logger.exception("Error while processing weather query")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process weather query: {exc}",
        )


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "weather_agent_api:app",
        host="0.0.0.0",
        port=port,
        reload=False,   # reload should be off in production
    )
