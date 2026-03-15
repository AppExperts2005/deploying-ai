"""
Service 3: Smart Calculator via Function Calling
Exposes two OpenAI function-calling tools:
  1. calculate(expression)     — evaluates math expressions safely
  2. convert_units(value, from_unit, to_unit) — converts between common units

All three services (weather, knowledge, calculator) are exposed as OpenAI
tool definitions in TOOLS so the LLM decides when to invoke each one.
"""

import math
import json
from typing import Any


# ---------------------------------------------------------------------------
# Safe math evaluator
# ---------------------------------------------------------------------------

_SAFE_MATH_NAMESPACE = {
    k: v for k, v in math.__dict__.items() if not k.startswith("_")
}
_SAFE_MATH_NAMESPACE.update({
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "pow": pow,
})


def calculate(expression: str) -> dict:
    """
    Safely evaluate a mathematical expression.
    Supports all math module functions (sqrt, sin, cos, log, etc.) plus
    basic arithmetic. Percentage syntax like '15% of 847' is pre-processed.

    Returns: {'result': value, 'expression': cleaned_expression}
             or {'error': message}
    """
    try:
        expr = expression.strip()

        # Handle "X% of Y" → X/100 * Y
        import re
        pct_match = re.match(r"([\d.]+)\s*%\s*of\s*([\d.]+)", expr, re.IGNORECASE)
        if pct_match:
            pct, total = float(pct_match.group(1)), float(pct_match.group(2))
            result = pct / 100 * total
            return {"result": round(result, 6), "expression": f"{pct}% of {total}"}

        # Replace ^ with ** for exponentiation
        expr = expr.replace("^", "**")

        result = eval(expr, {"__builtins__": {}}, _SAFE_MATH_NAMESPACE)  # noqa: S307

        if isinstance(result, (int, float)):
            return {"result": round(result, 8), "expression": expression}
        return {"error": f"Expression did not return a number: {result}"}

    except ZeroDivisionError:
        return {"error": "Division by zero."}
    except Exception as exc:
        return {"error": f"Could not evaluate '{expression}': {exc}"}


# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------

# Conversion tables: each value = multiplier to convert TO the base unit
_CONVERSIONS: dict[str, dict[str, float]] = {
    # Length — base: metre
    "length": {
        "m": 1, "metre": 1, "meter": 1, "metres": 1, "meters": 1,
        "km": 1000, "kilometre": 1000, "kilometer": 1000, "kilometres": 1000, "kilometers": 1000,
        "cm": 0.01, "centimetre": 0.01, "centimeter": 0.01,
        "mm": 0.001, "millimetre": 0.001, "millimeter": 0.001,
        "mi": 1609.344, "mile": 1609.344, "miles": 1609.344,
        "ft": 0.3048, "foot": 0.3048, "feet": 0.3048,
        "in": 0.0254, "inch": 0.0254, "inches": 0.0254,
        "yd": 0.9144, "yard": 0.9144, "yards": 0.9144,
        "nm": 1852, "nautical mile": 1852,
    },
    # Mass — base: kilogram
    "mass": {
        "kg": 1, "kilogram": 1, "kilograms": 1,
        "g": 0.001, "gram": 0.001, "grams": 0.001,
        "mg": 1e-6, "milligram": 1e-6,
        "lb": 0.453592, "lbs": 0.453592, "pound": 0.453592, "pounds": 0.453592,
        "oz": 0.0283495, "ounce": 0.0283495, "ounces": 0.0283495,
        "t": 1000, "tonne": 1000, "metric ton": 1000,
        "st": 6.35029, "stone": 6.35029,
    },
    # Volume — base: litre
    "volume": {
        "l": 1, "litre": 1, "liter": 1, "litres": 1, "liters": 1,
        "ml": 0.001, "millilitre": 0.001, "milliliter": 0.001,
        "cl": 0.01, "centilitre": 0.01,
        "gal": 3.78541, "gallon": 3.78541, "gallons": 3.78541,
        "qt": 0.946353, "quart": 0.946353, "quarts": 0.946353,
        "pt": 0.473176, "pint": 0.473176, "pints": 0.473176,
        "cup": 0.236588, "cups": 0.236588,
        "fl oz": 0.0295735, "fluid ounce": 0.0295735,
        "m3": 1000, "cubic meter": 1000, "cubic metre": 1000,
    },
    # Speed — base: m/s
    "speed": {
        "m/s": 1, "ms": 1, "metres per second": 1,
        "km/h": 1 / 3.6, "kmh": 1 / 3.6, "kph": 1 / 3.6, "kilometres per hour": 1 / 3.6,
        "mph": 0.44704, "miles per hour": 0.44704,
        "knot": 0.514444, "knots": 0.514444, "kn": 0.514444,
        "ft/s": 0.3048, "feet per second": 0.3048,
    },
    # Area — base: m²
    "area": {
        "m2": 1, "m²": 1, "square metre": 1, "square meter": 1,
        "km2": 1e6, "km²": 1e6, "square kilometre": 1e6, "square kilometer": 1e6,
        "cm2": 0.0001, "square centimetre": 0.0001,
        "ha": 10000, "hectare": 10000, "hectares": 10000,
        "acre": 4046.86, "acres": 4046.86,
        "ft2": 0.092903, "square foot": 0.092903, "square feet": 0.092903,
        "mi2": 2.59e6, "square mile": 2.59e6, "square miles": 2.59e6,
    },
    # Data — base: byte
    "data": {
        "b": 1, "byte": 1, "bytes": 1,
        "kb": 1024, "kilobyte": 1024, "kilobytes": 1024,
        "mb": 1048576, "megabyte": 1048576, "megabytes": 1048576,
        "gb": 1073741824, "gigabyte": 1073741824, "gigabytes": 1073741824,
        "tb": 1099511627776, "terabyte": 1099511627776, "terabytes": 1099511627776,
        "bit": 0.125, "bits": 0.125,
        "kib": 1024, "mib": 1048576, "gib": 1073741824,
    },
}

# Temperature gets special treatment (not multiplicative)
def _convert_temperature(value: float, from_unit: str, to_unit: str) -> float | None:
    def _normalise(unit: str) -> str:
        u = unit.lower().replace("°", "").replace("deg", "").strip()
        aliases = {
            "celsius": "c", "centigrade": "c",
            "fahrenheit": "f",
            "kelvin": "k",
            "rankine": "r",
        }
        return aliases.get(u, u[0] if u else u)

    f = _normalise(from_unit)
    t = _normalise(to_unit)

    valid = {"c", "f", "k", "r"}
    if f not in valid or t not in valid:
        return None

    # Convert to Celsius first
    if f == "c":   c = value
    elif f == "f": c = (value - 32) * 5 / 9
    elif f == "k": c = value - 273.15
    elif f == "r": c = (value - 491.67) * 5 / 9

    # Then to target
    if t == "c":   return c
    elif t == "f": return c * 9 / 5 + 32
    elif t == "k": return c + 273.15
    elif t == "r": return (c + 273.15) * 9 / 5


def convert_units(value: float, from_unit: str, to_unit: str) -> dict:
    """
    Convert a numeric value between units.
    Returns {'result': value, 'from': ..., 'to': ..., 'category': ...}
    or {'error': message}.
    """
    fu = from_unit.lower().strip()
    tu = to_unit.lower().strip()

    # Temperature (special case)
    temp_keywords = {"c", "f", "k", "r", "celsius", "fahrenheit", "kelvin",
                     "rankine", "centigrade", "°c", "°f", "°k", "degc", "degf"}
    if fu in temp_keywords or tu in temp_keywords:
        result = _convert_temperature(value, fu, tu)
        if result is None:
            return {"error": f"Cannot convert temperature from '{from_unit}' to '{to_unit}'."}
        return {
            "result":   round(result, 6),
            "from":     from_unit,
            "to":       to_unit,
            "category": "temperature",
        }

    # Search through conversion categories
    for category, table in _CONVERSIONS.items():
        if fu in table and tu in table:
            base_value = value * table[fu]    # convert to base unit
            result     = base_value / table[tu]  # convert to target
            return {
                "result":   round(result, 8),
                "from":     from_unit,
                "to":       to_unit,
                "category": category,
            }

    return {
        "error": (
            f"Cannot convert '{from_unit}' to '{to_unit}'. "
            "Supported categories: length, mass, volume, speed, area, temperature, data."
        )
    }


# ---------------------------------------------------------------------------
# OpenAI Tool definitions (all 3 services defined here for one import)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "Get the current real-time weather conditions for any city in the world. "
                "Use this whenever the user asks about weather, temperature, rain, forecast, "
                "humidity, wind, or climate conditions for a specific location."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The name of the city (e.g. 'Toronto', 'Paris', 'Tokyo').",
                    }
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Search a curated knowledge base of AI and machine learning concepts. "
                "Use this when the user asks about AI topics, ML algorithms, deep learning, "
                "NLP, model architectures, training techniques, or related research. "
                "Do NOT use for weather, calculations, or unrelated topics."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query about an AI/ML topic or concept.",
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of results to retrieve (default 3, max 5).",
                        "default": 3,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Evaluate a mathematical expression or perform a numeric calculation. "
                "Supports basic arithmetic (+, -, *, /), powers (^), percentages ('15% of 847'), "
                "and math functions: sqrt, sin, cos, tan, log, log10, exp, ceil, floor, etc. "
                "Use this whenever precise computation is needed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate, e.g. 'sqrt(144)' or '15% of 847'.",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "convert_units",
            "description": (
                "Convert a numeric value from one unit of measurement to another. "
                "Supports: length (km, miles, feet, inches…), mass (kg, lbs, oz…), "
                "volume (litres, gallons, cups…), speed (km/h, mph, m/s…), "
                "area (m², acres, hectares…), data (bytes, KB, MB, GB…), "
                "and temperature (Celsius, Fahrenheit, Kelvin)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {
                        "type": "number",
                        "description": "The numeric value to convert.",
                    },
                    "from_unit": {
                        "type": "string",
                        "description": "The unit to convert from (e.g. 'km', 'Celsius', 'lbs').",
                    },
                    "to_unit": {
                        "type": "string",
                        "description": "The unit to convert to (e.g. 'miles', 'Fahrenheit', 'kg').",
                    },
                },
                "required": ["value", "from_unit", "to_unit"],
            },
        },
    },
]


def dispatch_tool(name: str, arguments_json: str) -> Any:
    """Route a tool call by name and return the result as a Python object."""
    try:
        args = json.loads(arguments_json)
    except json.JSONDecodeError as exc:
        return {"error": f"Invalid tool arguments JSON: {exc}"}

    if name == "get_weather":
        from services.weather_service import get_weather
        return get_weather(args.get("city", ""))

    elif name == "search_knowledge_base":
        from services.knowledge_service import search_knowledge_base
        return search_knowledge_base(
            query=args.get("query", ""),
            n_results=args.get("n_results", 3),
        )

    elif name == "calculate":
        return calculate(args.get("expression", ""))

    elif name == "convert_units":
        return convert_units(
            value=args.get("value", 0),
            from_unit=args.get("from_unit", ""),
            to_unit=args.get("to_unit", ""),
        )

    else:
        return {"error": f"Unknown tool: {name}"}
