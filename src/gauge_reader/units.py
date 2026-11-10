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
    "inchesofwater": "pressure", "inchofwater": "pressure", "inwc": "pressure", "inwg": "pressure",
    "inchesh2o": "pressure", "inchh2o": "pressure", "\"h2o": "pressure", "inchesofwatercolumn": "pressure",
    "cmh2o": "pressure", "mh2o": "pressure",
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


# Pressure units -> pascals per unit (normalized spellings).
PA_PER_UNIT = {
    "pa": 1.0, "pascal": 1.0,
    "hpa": 100.0, "hectopascal": 100.0, "mbar": 100.0, "millibar": 100.0,
    "kpa": 1e3, "kilopascal": 1e3,
    "mpa": 1e6, "megapascal": 1e6,
    "bar": 1e5, "kbar": 1e8,
    "psi": 6894.757293, "psig": 6894.757293, "psia": 6894.757293, "lbf/in2": 6894.757293,
    "lb/in2": 6894.757293, "poundpersquareinch": 6894.757293, "poundspersquareinch": 6894.757293,
    "kgf/cm2": 98066.5, "kg/cm2": 98066.5, "kgf/cm^2": 98066.5, "kg/cm^2": 98066.5,
    "kilogram-forcepersquarecentimeter": 98066.5, "kilogramforcepersquarecentimeter": 98066.5,
    "atm": 101325.0, "atmosphere": 101325.0,
    "mmhg": 133.322387, "torr": 133.322368, "millimeterofmercury": 133.322387,
    "millimetersofmercury": 133.322387, "millimetreofmercury": 133.322387,
    "millimetresofmercury": 133.322387,
    "inhg": 3386.389, "inchofmercury": 3386.389, "inchesofmercury": 3386.389,
    "mmh2o": 9.80665, "mmaq": 9.80665, "mmwc": 9.80665, "cmh2o": 98.0665, "mh2o": 9806.65,
    "inh2o": 249.08891, "inchesofwater": 249.08891, "inchofwater": 249.08891, "inwc": 249.08891,
    "inwg": 249.08891, "inchesh2o": 249.08891, "inchh2o": 249.08891, "\"h2o": 249.08891,
    "inchesofwatercolumn": 249.08891,
}

_CELSIUS = {"°c", "℃", "degc", "c", "celsius", "degreecelsius"}
_FAHRENHEIT = {"°f", "℉", "degf", "f", "fahrenheit", "degreefahrenheit"}


def convert(value: float, src: str, dst: str) -> float | None:
    """Convert value from unit src to unit dst; None if not convertible."""
    a, b = norm_unit(src), norm_unit(dst)
    if a in PA_PER_UNIT and b in PA_PER_UNIT:
        return value * PA_PER_UNIT[a] / PA_PER_UNIT[b]
    if a in _CELSIUS | _FAHRENHEIT and b in _CELSIUS | _FAHRENHEIT:
        c = value if a in _CELSIUS else (value - 32.0) * 5.0 / 9.0
        return c if b in _CELSIUS else c * 9.0 / 5.0 + 32.0
    return None
