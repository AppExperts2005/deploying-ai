"""
Service 1: Real-Time Weather
Uses Open-Meteo free API (no API key required).
- Geocodes city names to lat/lon via Open-Meteo Geocoding API
- Retrieves current weather conditions and hourly forecast
- Returns structured data dict for LLM to narrate naturally
"""

import requests

# WMO Weather interpretation codes -> human-readable description
WMO_CODES = {
    0:  "clear sky",
    1:  "mainly clear",
    2:  "partly cloudy",
    3:  "overcast",
    45: "foggy",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    71: "slight snowfall",
    73: "moderate snowfall",
    75: "heavy snowfall",
    77: "snow grains",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    85: "slight snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with slight hail",
    99: "thunderstorm with heavy hail",
}

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL   = "https://api.open-meteo.com/v1/forecast"


def geocode_city(city: str) -> dict | None:
    """Return {name, country, latitude, longitude} for a city name, or None."""
    try:
        resp = requests.get(
            GEOCODING_URL,
            params={"name": city, "count": 1, "language": "en", "format": "json"},
            timeout=10,
        )
        resp.raise_for_status()
        results = resp.json().get("results")
        if not results:
            return None
        r = results[0]
        return {
            "name":      r.get("name", city),
            "country":   r.get("country", ""),
            "latitude":  r["latitude"],
            "longitude": r["longitude"],
            "timezone":  r.get("timezone", "UTC"),
        }
    except Exception as exc:
        print(f"[WeatherService] Geocoding error for '{city}': {exc}")
        return None


def get_weather(city: str) -> dict:
    """
    Fetch current weather for a city name.
    Returns a dict with weather data, or an error key.
    """
    location = geocode_city(city)
    if location is None:
        return {"error": f"Could not find location data for '{city}'."}

    try:
        resp = requests.get(
            WEATHER_URL,
            params={
                "latitude":               location["latitude"],
                "longitude":              location["longitude"],
                "current":                "temperature_2m,apparent_temperature,relative_humidity_2m,wind_speed_10m,weather_code,precipitation",
                "hourly":                 "temperature_2m,precipitation_probability",
                "forecast_days":          1,
                "wind_speed_unit":        "kmh",
                "temperature_unit":       "celsius",
                "timezone":               location["timezone"],
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        current = data.get("current", {})
        hourly  = data.get("hourly", {})

        weather_code = current.get("weather_code", 0)
        condition    = WMO_CODES.get(weather_code, "unknown conditions")

        # Next 6-hour precipitation probability average
        precip_probs = hourly.get("precipitation_probability", [])
        avg_precip   = round(sum(precip_probs[:6]) / max(len(precip_probs[:6]), 1), 1) if precip_probs else None

        return {
            "city":                  location["name"],
            "country":               location["country"],
            "temperature_c":         current.get("temperature_2m"),
            "feels_like_c":          current.get("apparent_temperature"),
            "humidity_pct":          current.get("relative_humidity_2m"),
            "wind_speed_kmh":        current.get("wind_speed_10m"),
            "condition":             condition,
            "precipitation_mm":      current.get("precipitation", 0),
            "precip_probability_pct": avg_precip,
        }

    except Exception as exc:
        print(f"[WeatherService] Weather fetch error: {exc}")
        return {"error": f"Failed to retrieve weather data: {exc}"}
