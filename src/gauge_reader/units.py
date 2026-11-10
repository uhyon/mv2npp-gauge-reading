"""Unit normalization shared by data filtering and evaluation."""

from __future__ import annotations

import re

# Normalized unit spellings (see norm_unit) -> quantity.
UNIT_KIND = {
    # pressure
    "bar": "pressure", "mbar": "pressure", "millibar": "pressure", "kbar": "pressure",
    "pa": "pressure", "pascal": "pressure", "hpa": "pressure", "hectopascal": "pressure",
    "kpa": "pressure", "kilopascal": "pressure", "mpa": "pressure", "megapascal": "pressure",
    "psi": "pressure", "psig": "pressure", "psia": "pressure", "lbf/in2": "pressure",
    "lb/in2": "pressure", "poundpersquareinch": "pressure", "poundspersquareinch": "pressure",
    "kgf/cm2": "pressure", "kg/cm2": "pressure", "kgf/cm^2": "pressure", "kg/cm^2": "pressure",
    "kilogram-forcepersquarecentimeter": "pressure", "kilogramforcepersquarecentimeter": "pressure",
    "atm": "pressure", "atmosphere": "pressure", "torr": "pressure",
    "mmhg": "pressure", "millimeterofmercury": "pressure", "millimetresofmercury": "pressure",
    "millimetersofmercury": "pressure", "millimetreofmercury": "pressure",
    "inhg": "pressure", "inchofmercury": "pressure", "inchesofmercury": "pressure",
    "mmh2o": "pressure", "mmaq": "pressure", "mmwc": "pressure", "inh2o": "pressure",
    "inchesofwater": "pressure", "cmh2o": "pressure", "mh2o": "pressure",
    # temperature
    "°c": "temperature", "℃": "temperature", "degc": "temperature", "c": "temperature",
    "celsius": "temperature", "degreecelsius": "temperature", "degreescelsius": "temperature",
    "°f": "temperature", "℉": "temperature", "degf": "temperature", "f": "temperature",
    "fahrenheit": "temperature", "degreefahrenheit": "temperature", "degreesfahrenheit": "temperature",
}


def norm_unit(u: str) -> str:
    s = u.strip().lower().replace("²", "2").replace("º", "°").replace("˚", "°")
    s = re.sub(r"[\s_.]+", "", s)
    s = s.replace("degrees", "degree")
    return s


def unit_kind(u) -> str | None:
    return UNIT_KIND.get(norm_unit(u)) if isinstance(u, str) else None


def flatten(x) -> list:
    """Flatten arbitrarily nested lists; a scalar becomes a one-item list."""
    if isinstance(x, (list, tuple)):
        return [v for item in x for v in flatten(item)]
    return [x]


def same_unit(a: str, b: str) -> bool:
    return isinstance(a, str) and isinstance(b, str) and norm_unit(a) == norm_unit(b)
