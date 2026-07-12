"""Tests for pure Home Assistant context classification."""

import sys
import types
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

# Avoid importing the integration adapter (and Home Assistant) for this pure
# module.  The package under test still uses its normal package-relative path.
# ruff: noqa: E402
_PACKAGE = "custom_components.adaptive_lighting"
_PACKAGE_PATH = (
    Path(__file__).resolve().parents[1] / "custom_components" / "adaptive_lighting"
)
if (package := sys.modules.get(_PACKAGE)) is None:
    package = types.ModuleType(_PACKAGE)
    package.__path__ = [str(_PACKAGE_PATH)]
    sys.modules[_PACKAGE] = package
elif str(_PACKAGE_PATH) not in package.__path__:
    package.__path__.append(str(_PACKAGE_PATH))

from custom_components.adaptive_lighting.context_classification import (
    ArrivalState,
    ContextClassification,
    HouseholdState,
    MediaState,
    OpeningState,
    SecurityState,
    SemanticIntent,
    StateSnapshot,
    WeatherDaylightState,
    classify_arrival_state,
    classify_context,
    classify_household_state,
    classify_media_state,
    classify_opening_state,
    classify_security_state,
    classify_weather_daylight,
)


@pytest.mark.parametrize(
    ("raw_state", "expected"),
    [
        ("triggered", SecurityState.EMERGENCY),
        ("alarm", SecurityState.EMERGENCY),
        ("fire", SecurityState.EMERGENCY),
        ("panic", SecurityState.EMERGENCY),
        ("emergency", SecurityState.EMERGENCY),
        ("armed_home", SecurityState.ARMED_HOME),
        ("armed_night", SecurityState.ARMED_NIGHT),
        ("armed_away", SecurityState.ARMED_AWAY),
        ("disarmed", SecurityState.DISARMED),
        ("problem", SecurityState.PROBLEM),
    ],
)
def test_security_states_are_explicit_and_deterministic(
    raw_state: str,
    expected: SecurityState,
) -> None:
    result = classify_security_state(
        {"state": raw_state, "entity_id": "alarm_control_panel.home"},
    )

    assert result.category is expected
    assert result.confidence > 0
    assert result.is_known is True
    assert result.provenance == ("state", "entity_id")


@pytest.mark.parametrize("raw_state", ["unknown", "unavailable", None, "", "armed"])
def test_security_unknown_and_ambiguous_states_fail_closed(raw_state: object) -> None:
    result = classify_security_state({"state": raw_state})

    assert result.category is SecurityState.UNKNOWN
    assert result.confidence == 0
    assert result.fails_closed is True
    assert result.semantic_intent is SemanticIntent.UNKNOWN


def test_security_alarm_attribute_outweighs_disarmed_state() -> None:
    result = classify_security_state(
        {
            "state": "disarmed",
            "attributes": {"alarm_state": "triggered"},
        },
    )

    assert result.category is SecurityState.EMERGENCY
    assert "attributes.alarm_state" in result.provenance
    assert "outranks" in result.reasons[0]
    assert result.semantic_intent is SemanticIntent.EMERGENCY


def test_safety_binary_sensor_on_is_an_emergency_but_generic_on_is_unknown() -> None:
    smoke = classify_security_state(
        {"state": "on", "attributes": {"device_class": "smoke"}},
    )
    generic = classify_security_state({"state": "on"})

    assert smoke.category is SecurityState.EMERGENCY
    assert generic.category is SecurityState.UNKNOWN


@pytest.mark.parametrize(
    ("raw_state", "expected"),
    [("on", SecurityState.EMERGENCY), ("off", SecurityState.DISARMED)],
)
def test_alarm_control_panel_on_and_off_are_classified_from_entity_id(
    raw_state: str,
    expected: SecurityState,
) -> None:
    result = classify_security_state(
        {"state": raw_state, "entity_id": "alarm_control_panel.home"},
    )

    assert result.category is expected


@pytest.mark.parametrize(
    ("content_type", "expected"),
    [
        ("video", MediaState.VIDEO),
        ("movie", MediaState.MOVIE),
        ("tvshow", MediaState.TV),
        ("music", MediaState.MUSIC),
        ("audio", MediaState.AUDIO),
        ("podcast", MediaState.PODCAST),
        ("game", MediaState.GAME),
    ],
)
def test_explicit_media_content_type_wins(
    content_type: str, expected: MediaState,
) -> None:
    result = classify_media_state(
        {
            "state": "playing",
            "attributes": {
                "media_content_type": content_type,
                "app_name": "Netflix",
            },
        },
    )

    assert result.category is expected
    assert result.confidence == 1.0
    assert result.semantic_intent in {
        SemanticIntent.MEDIA_VIDEO,
        SemanticIntent.MEDIA_AUDIO,
        SemanticIntent.MEDIA_GAME,
    }


@pytest.mark.parametrize(
    ("app_name", "expected"),
    [
        ("Netflix", MediaState.VIDEO),
        ("Plex", MediaState.VIDEO),
        ("YouTube", MediaState.VIDEO),
        ("Disney+", MediaState.VIDEO),
        ("Prime Video", MediaState.VIDEO),
        ("Apple TV", MediaState.VIDEO),
        ("Samsung TV", MediaState.VIDEO),
        ("Spotify", MediaState.MUSIC),
        ("Apple Music", MediaState.MUSIC),
        ("Apple Podcasts", MediaState.PODCAST),
        ("Steam", MediaState.GAME),
        ("PlayStation 5", MediaState.GAME),
        ("Xbox", MediaState.GAME),
    ],
)
def test_common_app_names_are_lower_confidence_fallbacks(
    app_name: str,
    expected: MediaState,
) -> None:
    result = classify_media_state(
        {"state": "playing", "attributes": {"app_name": app_name}},
    )

    assert result.category is expected
    assert result.confidence < 1.0
    assert "heuristic" in result.reasons[0]
    assert "attributes.app_name" in result.provenance


def test_explicit_content_type_wins_over_conflicting_app_id_and_name() -> None:
    result = classify_media_state(
        {
            "state": "playing",
            "attributes": {
                "media_content_type": "audio",
                "app_id": "com.netflix.mediaclient",
                "app_name": "Netflix",
                "title": "A movie",
            },
        },
    )

    assert result.category is MediaState.AUDIO
    assert result.confidence == 1.0
    assert "explicit content type outranks" in result.reasons[0]


def test_media_app_id_has_precedence_over_conflicting_app_name() -> None:
    result = classify_media_state(
        {
            "state": "playing",
            "attributes": {"app_id": "com.spotify.music", "app_name": "Netflix"},
        },
    )

    assert result.category is MediaState.MUSIC
    assert result.confidence == 0.85
    assert "attributes.app_id" in result.reasons[0]


def test_paused_media_is_not_idle() -> None:
    result = classify_media_state(
        {
            "state": "paused",
            "attributes": {"media_content_type": "movie"},
        },
    )

    assert result.category is MediaState.MOVIE
    assert result.category is not MediaState.IDLE


@pytest.mark.parametrize("raw_state", ["off", "idle", "standby"])
def test_idle_media_states_are_known_idle(raw_state: str) -> None:
    result = classify_media_state(
        {"state": raw_state, "attributes": {"app_name": "Netflix"}},
    )

    assert result.category is MediaState.IDLE
    assert result.confidence == 1.0
    assert result.semantic_intent is SemanticIntent.MEDIA_IDLE


@pytest.mark.parametrize("raw_state", ["unknown", "unavailable", "playing", None])
def test_media_without_reliable_state_or_type_fails_closed(raw_state: object) -> None:
    result = classify_media_state({"state": raw_state})

    if raw_state == "playing":
        assert result.category is MediaState.UNKNOWN
    else:
        assert result.category is MediaState.UNKNOWN
    assert result.confidence == 0
    assert result.fails_closed is True


@pytest.mark.parametrize(
    ("snapshot", "expected"),
    [
        ({"state": "on", "attributes": {"device_class": "door"}}, OpeningState.OPEN),
        (
            {"state": "off", "attributes": {"device_class": "window"}},
            OpeningState.CLOSED,
        ),
        ({"state": "open", "domain": "cover"}, OpeningState.OPEN),
        ({"state": "closed", "domain": "cover"}, OpeningState.CLOSED),
        ({"state": "opening", "domain": "cover"}, OpeningState.OPENING),
        ({"state": "closing", "domain": "cover"}, OpeningState.CLOSING),
        ({"state": "locked", "domain": "lock"}, OpeningState.LOCKED),
        ({"state": "unlocked", "domain": "lock"}, OpeningState.UNLOCKED),
        ({"state": "jammed", "domain": "cover"}, OpeningState.JAMMED),
    ],
)
def test_openings_are_classified_as_context_only(
    snapshot: dict[str, object],
    expected: OpeningState,
) -> None:
    result = classify_opening_state(snapshot)

    assert result.category is expected
    assert result.context_only is True
    assert result.confidence == 1.0
    assert result.semantic_intent in {
        SemanticIntent.OPENING_OPEN,
        SemanticIntent.OPENING_CLOSED,
        SemanticIntent.UNKNOWN,
    }


def test_generic_on_off_does_not_become_an_opening() -> None:
    result = classify_opening_state({"state": "on", "domain": "switch"})

    assert result.category is OpeningState.UNKNOWN
    assert result.fails_closed is True
    assert result.context_only is True


@pytest.mark.parametrize(
    ("weather_state", "expected"),
    [
        ("cloudy", WeatherDaylightState.CLOUDY),
        ("rainy", WeatherDaylightState.RAIN),
        ("lightning-rainy", WeatherDaylightState.STORM),
    ],
)
def test_weather_conditions_are_classified_before_daylight_proxies(
    weather_state: str,
    expected: WeatherDaylightState,
) -> None:
    result = classify_weather_daylight(
        {
            "state": weather_state,
            "attributes": {"sun_elevation": 25, "pv_power": 2500},
        },
    )

    assert result.category is expected
    assert result.context_only is True
    assert result.confidence > 0
    assert "state" in result.provenance


@pytest.mark.parametrize(
    ("elevation", "expected"),
    [
        (22, WeatherDaylightState.BRIGHT),
        (0, WeatherDaylightState.DIM),
        (-8, WeatherDaylightState.DARK),
    ],
)
def test_sun_elevation_produces_bright_dim_and_dark_context(
    elevation: float,
    expected: WeatherDaylightState,
) -> None:
    result = classify_weather_daylight({"sun_elevation": elevation})

    assert result.category is expected
    assert result.confidence == 0.95
    assert "attributes.sun_elevation" in result.provenance


def test_high_pv_is_only_a_daylight_proxy_and_zero_pv_does_not_prove_dark() -> None:
    high = classify_weather_daylight({"pv_power": 1800})
    direct_sensor = classify_weather_daylight(
        {"state": "900", "entity_id": "sensor.roof_pv_power"},
    )
    zero = classify_weather_daylight({"pv_power": 0})

    assert high.category is WeatherDaylightState.BRIGHT
    assert direct_sensor.category is WeatherDaylightState.BRIGHT
    assert direct_sensor.provenance == ("state", "entity_id")
    assert high.confidence == 0.65
    assert "daylight proxy" in high.reasons[0]
    assert "not lux or melanopic" in high.reasons[0]
    assert zero.category is WeatherDaylightState.UNKNOWN
    assert zero.fails_closed is True
    assert "does not prove darkness" in zero.reasons[0]


def test_unavailable_weather_daylight_values_fail_closed() -> None:
    result = classify_weather_daylight(
        {
            "state": "unavailable",
            "cloud_coverage": "unavailable",
            "sun_elevation": "unknown",
            "solar_irradiance": "unavailable",
            "pv_power": "unavailable",
        },
    )

    assert result.category is WeatherDaylightState.UNKNOWN
    assert result.confidence == 0
    assert result.fails_closed is True
    assert "attributes.pv_power" in result.provenance


@pytest.mark.parametrize(
    ("snapshot", "expected"),
    [
        ({"state": True}, HouseholdState.HOME),
        ({"state": False}, HouseholdState.AWAY),
        ({"state": "home"}, HouseholdState.HOME),
        ({"state": "not_home"}, HouseholdState.AWAY),
        ({"person_states": ["not_home", "home"]}, HouseholdState.HOME),
        ({"person_states": ["away", "not_home"]}, HouseholdState.AWAY),
        ({"state": ["not_home", "home"]}, HouseholdState.HOME),
        (
            {"persons": {"one": {"state": "home"}, "two": {"state": "away"}}},
            HouseholdState.HOME,
        ),
    ],
)
def test_household_state_uses_explicit_boolean_or_person_aggregates(
    snapshot: dict[str, object],
    expected: HouseholdState,
) -> None:
    result = classify_household_state(snapshot)

    assert result.category is expected
    assert result.is_known is True
    assert result.context_only is True


def test_incomplete_person_aggregate_fails_closed() -> None:
    result = classify_household_state(
        {"person_states": ["not_home", "unavailable"]},
    )

    assert result.category is HouseholdState.UNKNOWN
    assert result.fails_closed is True


def test_fresh_arrival_uses_bounded_age_and_exposes_provenance() -> None:
    result = classify_arrival_state(
        {
            "state": "arrived",
            "observed_at": 100,
            "age_seconds": 15,
            "max_age_seconds": 60,
        },
    )

    assert result.category is ArrivalState.RECENT
    assert result.confidence > 0.5
    assert result.semantic_intent is SemanticIntent.RECENT_ARRIVAL
    assert "attributes.observed_at" in result.provenance
    assert "attributes.max_age_seconds" in result.provenance


def test_arrival_age_can_be_derived_from_observed_at_and_explicit_now() -> None:
    result = classify_arrival_state(
        {"state": True},
        observed_at="2026-07-12T10:00:00+00:00",
        now="2026-07-12T10:00:30+00:00",
        max_age_seconds=60,
    )

    assert result.category is ArrivalState.RECENT
    assert "age 30s" in result.reasons[0]
    assert "argument.observed_at" in result.provenance


def test_stale_or_unbounded_arrival_fails_closed() -> None:
    stale = classify_arrival_state(
        {"state": True, "age_seconds": 61, "max_age_seconds": 60},
    )
    unbounded = classify_arrival_state({"state": True, "age_seconds": 1})

    assert stale.category is ArrivalState.STALE
    assert stale.confidence == 0
    assert stale.fails_closed is True
    assert "stale arrival fails closed" in stale.reasons[0]
    assert unbounded.category is ArrivalState.UNKNOWN
    assert unbounded.fails_closed is True


def test_home_presence_is_not_misclassified_as_sleep_or_arrival() -> None:
    result = classify_context(
        {
            "household": {"state": "home"},
            "sleep": {"state": "home"},
            "arrival": {"state": "home", "age_seconds": 1, "max_age_seconds": 60},
        },
    )

    assert result.household.category is HouseholdState.HOME
    assert result.sleep.fails_closed is True
    assert result.arrival.category is ArrivalState.UNKNOWN
    assert result.primary_semantic_intent is SemanticIntent.HOUSEHOLD_HOME


def test_combined_context_gives_emergency_precedence_over_media() -> None:
    result = classify_context(
        {
            "security": {"state": "triggered"},
            "sleep": {"state": True},
            "arrival": {
                "state": True,
                "age_seconds": 5,
                "max_age_seconds": 60,
            },
            "media": {
                "state": "playing",
                "attributes": {"media_content_type": "movie"},
            },
            "weather": {"state": "sunny"},
            "opening": {"state": "on", "attributes": {"device_class": "door"}},
        },
    )

    assert isinstance(result, ContextClassification)
    assert result.security.category is SecurityState.EMERGENCY
    assert result.media.category is MediaState.MOVIE
    assert result.opening.category is OpeningState.OPEN
    assert result.primary_semantic_intent is SemanticIntent.EMERGENCY
    assert result.semantic_intents == (SemanticIntent.EMERGENCY,)
    assert result.emergency is True
    assert "outranks media" in result.reasons[-1]


def test_combined_priority_is_sleep_security_arrival_media_then_weather() -> None:
    base = {
        "arrival": {"state": True, "age_seconds": 5, "max_age_seconds": 60},
        "media": {
            "state": "playing",
            "attributes": {"media_content_type": "movie"},
        },
        "weather": {"sun_elevation": 20},
    }
    arrival = classify_context(base)
    security = classify_context({**base, "security": {"state": "armed_home"}})
    sleep = classify_context(
        {**base, "security": {"state": "armed_home"}, "sleep": {"state": True}},
    )

    assert arrival.primary_semantic_intent is SemanticIntent.RECENT_ARRIVAL
    assert arrival.semantic_intents[:3] == (
        SemanticIntent.RECENT_ARRIVAL,
        SemanticIntent.MEDIA_VIDEO,
        SemanticIntent.DAYLIGHT_BRIGHT,
    )
    assert security.primary_semantic_intent is SemanticIntent.SECURITY_ARMED
    assert sleep.primary_semantic_intent is SemanticIntent.SLEEP


def test_mapping_and_dataclass_inputs_are_equivalent_and_results_are_immutable() -> (
    None
):
    mapping = {
        "state": "armed_away",
        "attributes": {"device_class": "alarm"},
        "entity_id": "alarm_control_panel.home",
    }
    from_mapping = classify_security_state(mapping)
    from_dataclass = classify_security_state(StateSnapshot(**mapping))

    assert from_mapping == from_dataclass
    with pytest.raises(FrozenInstanceError):
        from_mapping.confidence = 0.1  # type: ignore[misc]
