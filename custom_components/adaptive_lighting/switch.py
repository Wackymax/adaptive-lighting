"""Switch for the Adaptive Lighting integration."""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import math
import zoneinfo
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import homeassistant.helpers.config_validation as cv
import homeassistant.util.dt as dt_util
import ulid_transform
import voluptuous as vol
from homeassistant.components.light import (
    ATTR_BRIGHTNESS,
    ATTR_COLOR_TEMP_KELVIN,
    ATTR_RGB_COLOR,
    ATTR_SUPPORTED_COLOR_MODES,
    ATTR_TRANSITION,
    ATTR_XY_COLOR,
    ColorMode,
    LightEntityFeature,
    is_on,
    preprocess_turn_on_alternatives,
)
from homeassistant.components.light import DOMAIN as LIGHT_DOMAIN
from homeassistant.components.switch import DOMAIN as SWITCH_DOMAIN
from homeassistant.components.switch import SwitchEntity
from homeassistant.config_entries import SOURCE_IMPORT
from homeassistant.const import (
    ATTR_AREA_ID,
    ATTR_DOMAIN,
    ATTR_ENTITY_ID,
    ATTR_SERVICE,
    ATTR_SERVICE_DATA,
    ATTR_SUPPORTED_FEATURES,
    CONF_NAME,
    CONF_PARAMS,
    EVENT_CALL_SERVICE,
    EVENT_HOMEASSISTANT_STARTED,
    EVENT_STATE_CHANGED,
    SERVICE_TOGGLE,
    SERVICE_TURN_OFF,
    SERVICE_TURN_ON,
    STATE_OFF,
    STATE_ON,
)
from homeassistant.core import (
    CALLBACK_TYPE,
    Context,
    Event,
    HomeAssistant,
    ServiceCall,
    State,
    callback,
)
from homeassistant.helpers import entity_platform, entity_registry
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.entity_component import async_update_entity
from homeassistant.helpers.event import (
    EventStateChangedData,
    async_track_state_change_event,
    async_track_time_interval,
)
from homeassistant.helpers.restore_state import RestoreEntity
from homeassistant.util import slugify
from homeassistant.util.color import (
    color_temperature_to_rgb,
    color_xy_to_RGB,
)

from .adaptation_utils import (
    AdaptationData,
    LightControlAttributes,
    ServiceData,
    get_light_control_attributes,
    has_effect_attribute,
    manual_control_event_attribute_to_flags,
    prepare_adaptation_data,
)
from .behavior_runtime import BehaviorRuntimeAdapter, CandidateRecord
from .color_and_brightness import SunLightSettings
from .const import (
    ADAPT_BRIGHTNESS_SWITCH,
    ADAPT_COLOR_SWITCH,
    ATTR_ADAPT_BRIGHTNESS,
    ATTR_ADAPT_COLOR,
    ATTR_ADAPTIVE_LIGHTING_MANAGER,
    CONF_ADAPT_DELAY,
    CONF_ADAPT_ONLY_ON_BARE_TURN_ON,
    CONF_ADAPT_UNTIL_SLEEP,
    CONF_AMBIENT_BRIGHTNESS_CAP,
    CONF_AMBIENT_BRIGHTNESS_ENTITY,
    CONF_AUTORESET_CONTROL,
    CONF_BRIGHTNESS_MODE,
    CONF_BRIGHTNESS_MODE_TIME_DARK,
    CONF_BRIGHTNESS_MODE_TIME_LIGHT,
    CONF_DETECT_NON_HA_CHANGES,
    CONF_ENERGY_CONSTRAINT_ENTITY,
    CONF_HOME_STATE_ENTITY,
    CONF_ILLUMINANCE_ENTITIES,
    CONF_INCLUDE_CONFIG_IN_ATTRIBUTES,
    CONF_INITIAL_TRANSITION,
    CONF_INTELLIGENCE_AUTO_PROMOTE,
    CONF_INTELLIGENCE_DURABILITY_SECONDS,
    CONF_INTELLIGENCE_ENABLED,
    CONF_INTELLIGENCE_MINIMUM_CONFIDENCE,
    CONF_INTELLIGENCE_MINIMUM_SAMPLES,
    CONF_INTELLIGENCE_SHADOW_BASELINE_BRIGHTNESS,
    CONF_INTELLIGENCE_SHADOW_MODE,
    CONF_INTELLIGENCE_TRAINING_DAYS,
    CONF_INTELLIGENCE_TRAINING_ENABLED,
    CONF_INTERCEPT,
    CONF_INTERVAL,
    CONF_LIGHTS,
    CONF_MANUAL_CONTROL,
    CONF_MANUAL_HOLD_ENTITY,
    CONF_MAX_BRIGHTNESS,
    CONF_MAX_COLOR_TEMP,
    CONF_MAX_SUNRISE_TIME,
    CONF_MAX_SUNSET_TIME,
    CONF_MEDIA_ENTITIES,
    CONF_MIN_BRIGHTNESS,
    CONF_MIN_COLOR_TEMP,
    CONF_MIN_SUNRISE_TIME,
    CONF_MIN_SUNSET_TIME,
    CONF_MULTI_LIGHT_INTERCEPT,
    CONF_NIGHT_BRIGHTNESS_CAP,
    CONF_OCCUPANCY_ENTITIES,
    CONF_ONLY_ONCE,
    CONF_PREFER_RGB_COLOR,
    CONF_PRELIGHT_BRIGHTNESS_CAP,
    CONF_PRESENCE_ENTITIES,
    CONF_SECURITY_STATE_ENTITY,
    CONF_SEMANTIC_INTENT_ENTITY,
    CONF_SEND_SPLIT_DELAY,
    CONF_SEPARATE_TURN_ON_COMMANDS,
    CONF_SKIP_REDUNDANT_COMMANDS,
    CONF_SLEEP_BRIGHTNESS,
    CONF_SLEEP_COLOR_TEMP,
    CONF_SLEEP_ENTITY,
    CONF_SLEEP_RGB_COLOR,
    CONF_SLEEP_RGB_OR_COLOR_TEMP,
    CONF_SLEEP_TRANSITION,
    CONF_SUNRISE_OFFSET,
    CONF_SUNRISE_TIME,
    CONF_SUNSET_OFFSET,
    CONF_SUNSET_TIME,
    CONF_TAKE_OVER_CONTROL,
    CONF_TAKE_OVER_CONTROL_MODE,
    CONF_TASK_BRIGHTNESS_CAP,
    CONF_TRANSITION,
    CONF_TURN_ON_LIGHTS,
    CONF_USE_DEFAULTS,
    CONF_VIDEO_BRIGHTNESS_CAP,
    DOMAIN,
    EXTRA_VALIDATION,
    ICON_BRIGHTNESS,
    ICON_COLOR_TEMP,
    ICON_MAIN,
    ICON_SLEEP,
    INTELLIGENCE_SERVICE_SCHEMA,
    SERVICE_APPLY,
    SERVICE_CHANGE_SWITCH_SETTINGS,
    SERVICE_EXPLAIN,
    SERVICE_PREVIEW,
    SERVICE_SET_MANUAL_CONTROL,
    SET_MANUAL_CONTROL_SCHEMA,
    SLEEP_MODE_SWITCH,
    TURNING_OFF_DELAY,
    VALIDATION_TUPLES,
    TakeOverControlMode,
    apply_service_schema,
    replace_none_str,
)
from .context import ContextSignal, ContextSnapshot
from .context_classification import (
    ArrivalState,
    MediaState,
    SecurityState,
    classify_arrival_state,
    classify_household_state,
    classify_media_state,
    classify_security_state,
    classify_weather_daylight,
)
from .discovery import EntityDiscoveryCoordinator, InventorySnapshot
from .hass_utils import area_entities, setup_service_call_interceptor
from .helpers import (
    clamp,
    color_difference_redmean,
    int_to_base36,
    remove_vowels,
    short_hash,
)
from .policy import PolicyConfig, evaluate_policy
from .training import AdaptiveLightingTraining

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine, Iterable

    from homeassistant.config_entries import ConfigEntry
    from homeassistant.helpers.entity_platform import AddEntitiesCallback
    from homeassistant.helpers.typing import NoEventData, VolDictType

try:
    from homeassistant.helpers.sun import get_astral_observer
except ImportError:  # `get_astral_observer` was added in HA 2026.7
    from astral import Observer

    def get_astral_observer(hass: HomeAssistant) -> Observer:
        """Get an astral observer for the current HA configuration."""
        return Observer(
            hass.config.latitude,
            hass.config.longitude,
            hass.config.elevation,
        )


_LOGGER = logging.getLogger(__name__)

# Full discovery state remains in the coordinator and persisted learner. Only
# this compact, recent window is attached to the switch state because Recorder
# rejects attributes larger than 16 KiB and would otherwise log on every update
# in a large Home Assistant installation.
MAX_STATE_DIAGNOSTIC_ITEMS = 8
MAX_STATE_INTELLIGENCE_DECISIONS = 8
MAX_STATE_SAMPLE_COUNTS = 32
MAX_STATE_ATTRIBUTE_BYTES = 14_000


def _state_attribute_size(attributes: Mapping[str, Any]) -> int:
    """Estimate Recorder's compact JSON payload size for state attributes."""
    return len(
        json.dumps(attributes, default=str, separators=(",", ":")).encode("utf-8"),
    )


def _fit_state_attribute_budget(  # noqa: PLR0912 - ordered degradation stages
    attributes: dict[str, Any],
) -> dict[str, Any]:
    """Fail small when unusual user data would exceed Recorder's hard limit.

    Normal installations retain the complete bounded projection. This fallback
    affects state telemetry only; configuration, coordinator inventory, and
    learned models remain complete in their authoritative in-memory/storage
    locations.
    """
    if _state_attribute_size(attributes) <= MAX_STATE_ATTRIBUTE_BYTES:
        return attributes

    attributes["state_attributes_truncated"] = True
    attributes["configuration"] = {}
    attributes["configuration_truncated"] = True
    for key in ("manual_control",):
        value = attributes.get(key)
        if isinstance(value, list):
            attributes[key] = value[:MAX_STATE_DIAGNOSTIC_ITEMS]
    timers = attributes.get("autoreset_time_remaining")
    if isinstance(timers, Mapping):
        attributes["autoreset_time_remaining"] = dict(
            sorted(timers.items())[:MAX_STATE_DIAGNOSTIC_ITEMS],
        )

    decisions = attributes.get("intelligence_decisions")
    if isinstance(decisions, Mapping):
        compact_decisions: dict[str, dict[str, Any]] = {}
        decision_keys = (
            "intent",
            "baseline_brightness_pct",
            "target_brightness_pct",
            "confidence",
            "reason",
            "can_adjust",
            "can_turn_on",
            "can_turn_off",
        )
        for entity_id, decision in sorted(decisions.items())[
            :MAX_STATE_DIAGNOSTIC_ITEMS
        ]:
            if isinstance(decision, Mapping):
                compact_decisions[str(entity_id)] = {
                    key: decision[key] for key in decision_keys if key in decision
                }
        attributes["intelligence_decisions"] = compact_decisions
        attributes["intelligence_decisions_compacted"] = True

    discovery = attributes.get("intelligence_discovery")
    if isinstance(discovery, dict):
        discovery["areas"] = list(discovery.get("areas", []))[
            :MAX_STATE_DIAGNOSTIC_ITEMS
        ]
        for key in ("entities", "added", "removed", "moved", "renamed"):
            discovery[key] = list(discovery.get(key, []))[:MAX_STATE_DIAGNOSTIC_ITEMS]
        discovery["truncated"] = True

    if _state_attribute_size(attributes) <= MAX_STATE_ATTRIBUTE_BYTES:
        return attributes

    # Last-resort scalar summary guarantees a bounded state even with extreme
    # custom names or user-provided strings. Authoritative state is untouched.
    summary: dict[str, Any] = {
        "configuration": {},
        "configuration_truncated": True,
        "state_attributes_truncated": True,
    }
    for key, value in attributes.items():
        if isinstance(value, str):
            summary[key] = value[:256]
        elif value is None or isinstance(value, bool | int | float):
            summary[key] = value
    for key, allowed in {
        "intelligence": ("enabled", "configured_shadow_mode", "shadow_mode"),
        "intelligence_training": (
            "active",
            "phase",
            "training_deadline",
            "remaining_seconds",
            "confidence",
            "promotion_reason",
        ),
        "intelligence_behavior": (
            "phase",
            "models",
            "candidates",
            "available_candidates",
            "accepted_observations",
            "pending",
            "corrections",
            "suppression",
            "shadow_ready",
            "shadow_executed",
            "actuation_enabled",
        ),
        "intelligence_discovery": ("revision", "reason", "entity_count"),
    }.items():
        value = attributes.get(key)
        if not isinstance(value, Mapping):
            continue
        summary[key] = {
            item: (value[item][:256] if isinstance(value[item], str) else value[item])
            for item in allowed
            if item in value
            and (
                isinstance(value[item], str | bool | int | float) or value[item] is None
            )
        }
    return summary


SCAN_INTERVAL = timedelta(seconds=10)

# Consider it a significant change when attribute changes more than
BRIGHTNESS_CHANGE = 25  # ≈10% of total range
COLOR_TEMP_CHANGE = 100  # ≈3% of total range (2000-6500)
RGB_REDMEAN_CHANGE = 80  # ≈10% of total range


# Keep a short domain version for the context instances (which can only be 36 chars)
_DOMAIN_SHORT = "al"


def _state_is_active(state: State | None) -> bool:
    """Return a conservative boolean interpretation for context entities."""
    if state is None:
        return False
    value = state.state.strip().lower()
    return value in {
        "on",
        "active",
        "home",
        "occupied",
        "present",
        "playing",
        "true",
        "yes",
        "1",
    }


def _json_safe(value: Any) -> Any:
    """Convert a policy result into Home Assistant state-attribute data."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if is_dataclass(value):
        return _json_safe(asdict(value))
    enum_value = getattr(value, "value", None)
    if enum_value is not None and enum_value is not value:
        return _json_safe(enum_value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _evaluate_intelligence_policy(
    snapshot_values: dict[str, Any],
    policy_values: dict[str, Any],
    intent_hint: str,
) -> dict[str, Any] | None:
    """Evaluate the pure policy through one explicit Home Assistant adapter.

    This boundary is deliberately typed. If the pure API changes, evaluation
    fails safe to the deterministic baseline instead of silently guessing a
    constructor shape and producing a plausible but incorrect decision.
    """
    try:
        raw_signals = snapshot_values["signals"]

        def source_for(name: str) -> str:
            for signal in raw_signals:
                if signal["name"] == name:
                    return signal["source"]
            return "adaptive_lighting"

        def make_signal(
            name: str,
            value: Any,
            available: bool = True,
        ) -> ContextSignal[Any]:
            return ContextSignal(
                value=value if available else None,
                source=source_for(name),
                available=available,
                confidence=1.0 if available else 0.0,
                detail=name,
            )

        def boolean_signal(name: str, key: str) -> ContextSignal[bool]:
            available = bool(snapshot_values.get(f"{key}_available", False))
            return make_signal(name, bool(snapshot_values.get(key)), available)

        semantic = snapshot_values.get("semantic_intent") or ""
        snapshot = ContextSnapshot(
            emergency=make_signal(
                "security_state_entity",
                bool(snapshot_values.get("emergency")),
                bool(snapshot_values.get("emergency_available", False)),
            ),
            manual=make_signal(
                "manual_hold",
                bool(snapshot_values.get("manual_control")),
                bool(snapshot_values.get("manual_control")),
            ),
            manual_hold=make_signal(
                "manual_hold",
                bool(snapshot_values.get("manual_hold")),
                bool(snapshot_values.get("manual_hold_available", False)),
            ),
            sleep=make_signal(
                "sleep_entity",
                bool(snapshot_values.get("sleep")),
                bool(snapshot_values.get("sleep_available", False)),
            ),
            sleep_mode=make_signal(
                "sleep_entity",
                bool(snapshot_values.get("sleep")),
                bool(snapshot_values.get("sleep_available", False)),
            ),
            night_path=make_signal(
                "semantic_intent_entity",
                intent_hint == "night",
                "night" in semantic,
            ),
            task=make_signal(
                "semantic_intent_entity",
                intent_hint == "task",
                any(token in semantic for token in ("task", "focus", "work")),
            ),
            video=make_signal(
                "media_entities",
                intent_hint == "video",
                bool(snapshot_values.get("media_available", False))
                or any(token in semantic for token in ("video", "movie", "cinema")),
            ),
            video_playing=make_signal(
                "media_entities",
                bool(snapshot_values.get("video_playing")),
                bool(snapshot_values.get("media_available", False)),
            ),
            arrival=make_signal(
                "semantic_intent_entity",
                intent_hint == "prelight",
                "prelight" in semantic,
            ),
            ambient=make_signal(
                "semantic_intent_entity",
                intent_hint == "ambient",
                bool(snapshot_values.get("semantic_intent_available", False)),
            ),
            ambient_brightness=make_signal(
                "ambient_brightness_entity",
                snapshot_values.get("ambient_brightness_value"),
                bool(snapshot_values.get("ambient_brightness_available", False)),
            ),
            vacant=make_signal(
                "occupancy_entities",
                not bool(snapshot_values.get("occupancy")),
                bool(snapshot_values.get("occupancy_available", False)),
            ),
            occupancy=boolean_signal("occupancy_entities", "occupancy"),
            illuminance=make_signal(
                "illuminance_entities",
                snapshot_values.get("illuminance_value"),
                bool(snapshot_values.get("illuminance_available", False)),
            ),
            requested_brightness=make_signal(
                "adaptive_lighting",
                policy_values["baseline_brightness_pct"],
            ),
            current_brightness=make_signal(
                "adaptive_lighting",
                policy_values["current_brightness_pct"],
            ),
            semantic_intent=make_signal(
                "semantic_intent_entity",
                semantic,
                bool(snapshot_values.get("semantic_intent_available", False))
                and bool(semantic),
            ),
            intent_hint=make_signal(
                "adaptive_lighting",
                intent_hint,
                intent_hint != "adaptive",
            ),
        )
        caps = policy_values
        policy_config = PolicyConfig(
            min_brightness=0.0,
            max_brightness=caps["baseline_brightness_pct"],
            emergency_brightness=caps["baseline_brightness_pct"],
            task_brightness=caps["task"],
            video_brightness=caps["video"],
            arrival_brightness=caps["prelight"],
            ambient_brightness=caps["ambient"],
            vacant_brightness=0.0,
            sleep_brightness=caps["night"],
            sleep_max_brightness=caps["night"],
            night_path_brightness=caps["night"],
            night_path_max_brightness=caps["night"],
        )
        decision = evaluate_policy(snapshot, policy_config)
    except (TypeError, ValueError, KeyError, AttributeError):
        _LOGGER.warning(
            "Adaptive Lighting intelligence policy evaluation failed; "
            "using the conservative runtime fallback",
            exc_info=True,
        )
        return None

    return {
        "intent": _json_safe(decision.intent),
        "target_brightness_pct": decision.brightness_target,
        "confidence": decision.confidence,
        "reason": " | ".join(decision.reasons),
        # State attributes are recorder data. Keep only the bounded set of
        # usable inputs instead of duplicating every unavailable schema field.
        "provenance": _json_safe(
            tuple(item for item in decision.input_provenance if item.available)[:12],
        ),
        "can_adjust": decision.can_adjust,
        "can_turn_on": decision.can_turn_on,
        "can_turn_off": decision.can_turn_off,
    }


def create_context(
    name: str,
    which: str,
    index: int,
    parent: Context | None = None,
) -> Context:
    """Create a context that can identify this integration."""
    # Use a hash for the name because otherwise the context might become
    # too long (max len == 26) to fit in the database.
    # Pack index with base85 to maximize the number of contexts we can create
    # before we exceed the 26-character limit and are forced to wrap.
    time_stamp = ulid_transform.ulid_now()[:10]  # time part of a ULID
    name_hash = short_hash(name)
    which_short = remove_vowels(which)
    context_id_start = f"{time_stamp}:{_DOMAIN_SHORT}:{name_hash}:{which_short}:"
    chars_left = 26 - len(context_id_start)
    index_packed = int_to_base36(index).zfill(chars_left)[-chars_left:]
    context_id = context_id_start + index_packed
    parent_id = parent.id if parent else None
    return Context(id=context_id, parent_id=parent_id)


def is_our_context_id(context_id: str | None, which: str | None = None) -> bool:
    """Check whether this integration created 'context_id'."""
    if context_id is None:
        return False

    is_al = f":{_DOMAIN_SHORT}:" in context_id
    if not is_al:
        return False
    if which is None:
        return True
    return f":{remove_vowels(which)}:" in context_id


def is_our_context(context: Context | None, which: str | None = None) -> bool:
    """Check whether this integration created 'context'."""
    if context is None:
        return False
    return is_our_context_id(context.id, which)


def _switches_with_lights(
    hass: HomeAssistant,
    lights: list[str],
    expand_light_groups: bool = True,
) -> AdaptiveSwitches:
    """Get all switches that control at least one of the lights passed."""
    config_entries = hass.config_entries.async_entries(DOMAIN)
    data = hass.data[DOMAIN]
    switches: AdaptiveSwitches = []
    all_check_lights = (
        _expand_light_groups(hass, lights) if expand_light_groups else set(lights)
    )
    for config in config_entries:
        entry = data.get(config.entry_id)
        if entry is None:  # entry might be disabled and therefore missing
            continue
        switch = data[config.entry_id][SWITCH_DOMAIN]
        switch._expand_light_groups(hass=hass)
        # Check if any of the lights are in the switch's lights
        if set(switch.lights) & set(all_check_lights):
            switches.append(switch)
    return switches


class NoSwitchFoundError(ValueError):
    """No switches found for lights."""


def _switch_with_lights(
    hass: HomeAssistant,
    lights: list[str],
    expand_light_groups: bool = True,
) -> AdaptiveSwitch:
    """Find the switch that controls the lights in 'lights'."""
    switches = _switches_with_lights(hass, lights, expand_light_groups)
    if len(switches) == 1:
        return switches[0]
    if len(switches) > 1:
        on_switches = [s for s in switches if s.is_on]
        if len(on_switches) == 1:
            # Of the multiple switches, only one is on
            return on_switches[0]
        msg = (
            f"_switch_with_lights: Light(s) {lights} found in multiple switch configs"
            f" ({[s.entity_id for s in switches]}). You must pass a switch under"
            " 'entity_id'."
        )
        raise NoSwitchFoundError(msg)
    msg = (
        f"_switch_with_lights: Light(s) {lights} not found in any switch's"
        " configuration. You must either include the light(s) that is/are"
        " in the integration config, or pass a switch under 'entity_id'."
    )
    raise NoSwitchFoundError(msg)


# For documentation on this function, see integration_entities() from HomeAssistant Core:
# https://github.com/home-assistant/core/blob/dev/homeassistant/helpers/template.py#L1109
def _switches_from_service_call(
    hass: HomeAssistant,
    service_call: ServiceCall,
) -> AdaptiveSwitches:
    data = service_call.data
    lights = data[CONF_LIGHTS]
    switch_entity_ids: list[str] | None = data.get("entity_id")

    if not lights and not switch_entity_ids:
        msg = (
            "adaptive-lighting: Neither a switch nor a light was provided in the service call."
            " If you intend to adapt all lights on all switches, please inform the"
            " developers at https://github.com/basnijholt/adaptive-lighting about your"
            " use case. Currently, you must pass either an adaptive-lighting switch or"
            " the lights to an `adaptive_lighting` service call."
        )
        raise ValueError(msg)

    if switch_entity_ids is not None:
        if len(switch_entity_ids) > 1 and lights:
            msg = (
                "adaptive-lighting: Cannot pass multiple switches with lights argument."
                f" Invalid service data received: {service_call.data}"
            )
            raise ValueError(msg)
        switches: AdaptiveSwitches = []
        ent_reg = entity_registry.async_get(hass)
        for entity_id in switch_entity_ids:
            ent_entry = ent_reg.async_get(entity_id)
            assert ent_entry is not None
            config_id = ent_entry.config_entry_id
            switches.append(hass.data[DOMAIN][config_id][SWITCH_DOMAIN])
        return switches

    if lights:
        switch = _switch_with_lights(hass, lights)
        return [switch]

    msg = (
        "adaptive-lighting: Incorrect data provided in service call."
        f" Entities not found in the integration. Service data: {service_call.data}"
    )
    raise ValueError(msg)


async def handle_change_switch_settings(
    switch: AdaptiveSwitch,
    service_call: ServiceCall,
) -> None:
    """Allows HASS to change config values via a service call."""
    data = service_call.data
    which = data.get(CONF_USE_DEFAULTS, "current")
    if which == "current":  # use whatever we're already using.
        defaults = switch._current_settings  # pylint: disable=protected-access
    elif which == "factory":  # use actual defaults listed in the documentation
        defaults = None
    elif which == "configuration":
        # use whatever's in the config flow or configuration.yaml
        defaults = switch._config_backup
    else:
        defaults = None

    # deep copy the defaults so we don't modify the original dicts
    switch._set_changeable_settings(data=data, defaults=deepcopy(defaults))
    if switch.is_on:
        switch._update_time_interval_listener()

    _LOGGER.debug(
        "Called 'adaptive_lighting.change_switch_settings' service with '%s'",
        data,
    )

    switch.manager.reset(*switch.lights, reset_manual_control=False)
    if switch.is_on:
        await switch._update_attrs_and_maybe_adapt_lights(  # pylint: disable=protected-access
            context=switch.create_context("service", parent=service_call.context),
            lights=switch.lights,
            transition=switch.initial_transition,
            force=True,
        )


async def async_setup_entry(  # noqa: PLR0915
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the AdaptiveLighting switch."""
    assert hass is not None
    data = hass.data[DOMAIN]
    assert config_entry.entry_id in data
    _LOGGER.debug(
        "Setting up AdaptiveLighting with data: %s and config_entry %s",
        data,
        config_entry,
    )
    if (  # Skip deleted YAML config entries or first time YAML config entries
        config_entry.source == SOURCE_IMPORT
        and config_entry.unique_id not in data.get("__yaml__", set())
    ):
        _LOGGER.warning(
            "Deleting AdaptiveLighting switch '%s' because YAML"
            " defined switch has been removed from YAML configuration",
            config_entry.unique_id,
        )
        await hass.config_entries.async_remove(config_entry.entry_id)
        return

    if (manager := data.get(ATTR_ADAPTIVE_LIGHTING_MANAGER)) is None:
        manager = AdaptiveLightingManager(hass)
        data[ATTR_ADAPTIVE_LIGHTING_MANAGER] = manager

    entry_settings = validate(config_entry)
    intelligence_shadow = bool(
        entry_settings[CONF_INTELLIGENCE_ENABLED]
        and entry_settings[CONF_INTELLIGENCE_SHADOW_MODE],
    )
    sleep_mode_switch = SimpleSwitch(
        which="Sleep Mode",
        initial_state=False,
        hass=hass,
        config_entry=config_entry,
        icon=ICON_SLEEP,
    )
    adapt_color_switch = SimpleSwitch(
        which="Adapt Color",
        # A stale restore record must not re-enable deterministic color
        # adaptation while the intelligence layer is commissioned in shadow.
        initial_state=not intelligence_shadow,
        restore_state=not intelligence_shadow,
        hass=hass,
        config_entry=config_entry,
        icon=ICON_COLOR_TEMP,
    )
    adapt_brightness_switch = SimpleSwitch(
        which="Adapt Brightness",
        initial_state=True,
        hass=hass,
        config_entry=config_entry,
        icon=ICON_BRIGHTNESS,
    )
    switch = AdaptiveSwitch(
        hass,
        config_entry,
        manager,
        sleep_mode_switch,
        adapt_color_switch,
        adapt_brightness_switch,
    )

    data[config_entry.entry_id][SLEEP_MODE_SWITCH] = sleep_mode_switch
    data[config_entry.entry_id][ADAPT_COLOR_SWITCH] = adapt_color_switch
    data[config_entry.entry_id][ADAPT_BRIGHTNESS_SWITCH] = adapt_brightness_switch
    data[config_entry.entry_id][SWITCH_DOMAIN] = switch

    async_add_entities(
        [sleep_mode_switch, adapt_color_switch, adapt_brightness_switch, switch],
        update_before_add=True,
    )

    @callback
    async def handle_apply(service_call: ServiceCall) -> None:
        """Handle the entity service apply."""
        data = service_call.data
        _LOGGER.debug(
            "Called 'adaptive_lighting.apply' service with '%s'",
            data,
        )
        switches = _switches_from_service_call(hass, service_call)
        lights = data[CONF_LIGHTS]
        for switch in switches:
            if not lights:
                all_lights = switch.lights
            else:
                all_lights = _expand_light_groups(hass, lights)
            switch.manager.lights.update(all_lights)
            for light in all_lights:
                if data[CONF_TURN_ON_LIGHTS] or is_on(hass, light):
                    context = switch.create_context(
                        "service",
                        parent=service_call.context,
                    )
                    await switch._adapt_light(  # pylint: disable=protected-access
                        light,
                        context=context,
                        transition=data[CONF_TRANSITION],
                        adapt_brightness=data[ATTR_ADAPT_BRIGHTNESS],
                        adapt_color=data[ATTR_ADAPT_COLOR],
                        prefer_rgb_color=data[CONF_PREFER_RGB_COLOR],
                        force=True,
                    )

    @callback
    async def handle_set_manual_control(service_call: ServiceCall) -> None:
        """Set or unset lights as 'manually controlled'."""
        data = service_call.data
        _LOGGER.debug(
            "Called 'adaptive_lighting.set_manual_control' service with '%s'",
            data,
        )
        switches = _switches_from_service_call(hass, service_call)
        lights = data[CONF_LIGHTS]
        for switch in switches:
            if not lights:
                all_lights = switch.lights
            else:
                all_lights = _expand_light_groups(hass, lights)

            manual_attributes = manual_control_event_attribute_to_flags(
                service_call.data[CONF_MANUAL_CONTROL],
            )

            if manual_attributes:
                for light in all_lights:
                    switch.manager.set_manual_control_attributes(
                        light,
                        manual_attributes,
                    )
                    switch.fire_manual_control_event(
                        light,
                        service_call.context,
                    )
            else:
                switch.manager.reset(*all_lights)
                if switch.is_on:
                    context = switch.create_context(
                        "service",
                        parent=service_call.context,
                    )
                    # pylint: disable=protected-access
                    await switch._update_attrs_and_maybe_adapt_lights(
                        context=context,
                        lights=all_lights,
                        transition=switch.initial_transition,
                        force=True,
                    )

    @callback
    async def handle_intelligence_service(service_call: ServiceCall) -> None:
        """Publish a read-only intelligence preview/explanation event."""
        switches = _switches_from_service_call(hass, service_call)
        lights = service_call.data[CONF_LIGHTS]
        for switch in switches:
            selected_lights = (
                switch.lights
                if not lights
                else _expand_light_groups(
                    hass,
                    lights,
                )
            )
            decisions = switch.preview_intelligence(selected_lights)
            hass.bus.async_fire(
                f"{DOMAIN}.intelligence_{service_call.service}",
                {
                    ATTR_ENTITY_ID: switch.entity_id,
                    "decisions": decisions,
                    "shadow_mode": switch._intelligence_shadow_actuation_blocked,
                },
                context=service_call.context,
            )

    # Register `apply` service
    hass.services.async_register(
        domain=DOMAIN,
        service=SERVICE_APPLY,
        service_func=handle_apply,
        schema=apply_service_schema(switch.initial_transition),
    )

    # Register `set_manual_control` service
    hass.services.async_register(
        domain=DOMAIN,
        service=SERVICE_SET_MANUAL_CONTROL,
        service_func=handle_set_manual_control,
        schema=SET_MANUAL_CONTROL_SCHEMA,
    )

    for service in (SERVICE_PREVIEW, SERVICE_EXPLAIN):
        hass.services.async_register(
            domain=DOMAIN,
            service=service,
            service_func=handle_intelligence_service,
            schema=INTELLIGENCE_SERVICE_SCHEMA,
        )

    args: VolDictType = {vol.Optional(CONF_USE_DEFAULTS, default="current"): cv.string}
    # Modifying these after init isn't possible
    skip = (CONF_INTERVAL, CONF_NAME, CONF_LIGHTS)
    for k, _, valid in VALIDATION_TUPLES:
        if k not in skip:
            args[vol.Optional(k)] = valid
    platform = entity_platform.current_platform.get()
    assert platform is not None
    platform.async_register_entity_service(
        SERVICE_CHANGE_SWITCH_SETTINGS,
        args,
        handle_change_switch_settings,
    )


def validate(
    config_entry: ConfigEntry | None,
    service_data: dict[str, Any] | None = None,
    defaults: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Get the options and data from the config_entry and add defaults."""
    if defaults is None:
        data = {key: default for key, default, _ in VALIDATION_TUPLES}
    else:
        data = deepcopy(defaults)

    if config_entry is not None:
        assert service_data is None
        assert defaults is None
        data.update(config_entry.options)  # come from options flow
        data.update(config_entry.data)  # all yaml settings come from data
    else:
        assert service_data is not None
        changed_settings = {
            key: value
            for key, value in service_data.items()
            if key not in (CONF_USE_DEFAULTS, ATTR_ENTITY_ID)
        }
        data.update(changed_settings)
    data = {key: replace_none_str(value) for key, value in data.items()}
    for key, (validate_value, _) in EXTRA_VALIDATION.items():
        value = data.get(key)
        if value is not None:
            data[key] = validate_value(value)  # Fix the types of the inputs
    return data


def _is_state_event(
    event: Event[EventStateChangedData],
    from_or_to_state: Iterable[str],
) -> bool:
    """Match state event when either 'from_state' or 'to_state' matches."""
    return (
        (old_state := event.data.get("old_state")) is not None
        and old_state.state in from_or_to_state
    ) or (
        (new_state := event.data.get("new_state")) is not None
        and new_state.state in from_or_to_state
    )


def _expand_light_groups(
    hass: HomeAssistant,
    lights: list[str],
) -> list[str]:
    all_lights: set[str] = set()
    manager = hass.data[DOMAIN][ATTR_ADAPTIVE_LIGHTING_MANAGER]
    for light in lights:
        state = hass.states.get(light)
        if state is None:
            _LOGGER.debug("State of %s is None", light)
            all_lights.add(light)
        elif _is_light_group(state):
            group = state.attributes["entity_id"]
            manager.lights.discard(light)
            all_lights.update(group)
            _LOGGER.debug("Expanded %s to %s", light, group)
        else:
            all_lights.add(light)
    return sorted(all_lights)


def _is_light_group(state: State) -> bool:
    return "entity_id" in state.attributes and not state.attributes.get(
        "is_hue_group",
        False,
    )


def _supported_features(hass: HomeAssistant, light: str) -> set[str]:
    state = hass.states.get(light)
    assert state is not None
    supported_features = int(
        state.attributes.get(ATTR_SUPPORTED_FEATURES, 0),
    )  # type: ignore[arg-type]
    assert isinstance(supported_features, int)

    supported: set[str] = set()

    if supported_features & LightEntityFeature.TRANSITION:
        supported.add("transition")

    supported_color_modes = state.attributes.get(
        ATTR_SUPPORTED_COLOR_MODES,
        set(),
    )  # type: ignore[arg-type]
    color_modes = {
        ColorMode.RGB,
        ColorMode.RGBW,
        ColorMode.RGBWW,
        ColorMode.XY,
        ColorMode.HS,
    }

    # Adding brightness when color mode is supported, see
    # comment https://github.com/basnijholt/adaptive-lighting/issues/112#issuecomment-836944011

    for mode in color_modes:
        if mode in supported_color_modes:
            supported.update({"color", "brightness"})
            break

    if ColorMode.COLOR_TEMP in supported_color_modes:
        supported.update({"color_temp", "brightness"})

    if ColorMode.BRIGHTNESS in supported_color_modes:
        supported.add("brightness")

    return supported


# All comparisons should be done with RGB since
# converting anything to color temp is inaccurate.
def _convert_attributes(attributes: dict[str, Any]) -> dict[str, Any]:
    if ATTR_RGB_COLOR in attributes:
        return attributes

    rgb = None
    if (color := attributes.get(ATTR_COLOR_TEMP_KELVIN)) is not None:
        rgb = color_temperature_to_rgb(color)
    elif (color := attributes.get(ATTR_XY_COLOR)) is not None:
        rgb = color_xy_to_RGB(*color)

    if rgb is not None:
        attributes[ATTR_RGB_COLOR] = rgb
        _LOGGER.debug("Converted attributes %s to rgb %s", attributes, rgb)
    else:
        _LOGGER.debug("No suitable color conversion found for %s", attributes)

    return attributes


def _add_missing_attributes(
    old_attributes: dict[str, Any],
    new_attributes: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not any(
        attr in old_attributes and attr in new_attributes
        for attr in [ATTR_COLOR_TEMP_KELVIN, ATTR_RGB_COLOR]
    ):
        old_attributes = _convert_attributes(old_attributes)
        new_attributes = _convert_attributes(new_attributes)

    return old_attributes, new_attributes


def _has_color_mode_changed(
    light: str,
    old_attributes: dict[str, Any],
    new_attributes: dict[str, Any],
    context: Context,
) -> bool:
    """Check if the light's color mode changed (e.g., color_temp to RGB or vice versa).

    This must be called BEFORE _add_missing_attributes() to detect mode changes
    using the original attributes. See issue #1275.
    """
    old_has_color_temp = old_attributes.get(ATTR_COLOR_TEMP_KELVIN) is not None
    old_has_rgb = old_attributes.get(ATTR_RGB_COLOR) is not None
    old_has_xy = old_attributes.get(ATTR_XY_COLOR) is not None

    new_has_color_temp = new_attributes.get(ATTR_COLOR_TEMP_KELVIN) is not None
    new_has_rgb = new_attributes.get(ATTR_RGB_COLOR) is not None
    new_has_xy = new_attributes.get(ATTR_XY_COLOR) is not None

    # Determine old and new color modes
    # Priority: color_temp > rgb > xy (matching typical light behavior)
    if old_has_color_temp:
        old_mode = "color_temp"
    elif old_has_rgb:
        old_mode = "rgb"
    elif old_has_xy:
        old_mode = "xy"
    else:
        old_mode = None

    if new_has_color_temp:
        new_mode = "color_temp"
    elif new_has_rgb:
        new_mode = "rgb"
    elif new_has_xy:
        new_mode = "xy"
    else:
        new_mode = None

    # Check if mode changed
    if old_mode is not None and new_mode is not None and old_mode != new_mode:
        _LOGGER.debug(
            "Light mode of %s changed from %s to %s with context.id='%s'",
            light,
            old_mode,
            new_mode,
            context.id,
        )
        return True
    return False


def _attributes_have_changed(
    light: str,
    old_attributes: dict[str, Any],
    new_attributes: dict[str, Any],
    context: Context,
) -> LightControlAttributes:
    # 2023-11-19: HA core no longer removes light domain attributes when off
    # so we must protect for `None` here
    # see https://github.com/home-assistant/core/pull/101946

    changed_attributes = LightControlAttributes.NONE

    # Check for color mode changes BEFORE attribute conversion
    # This detects external changes like Hue scenes switching from color_temp to RGB
    # See: https://github.com/basnijholt/adaptive-lighting/issues/1275
    if _has_color_mode_changed(
        light,
        old_attributes,
        new_attributes,
        context,
    ):
        changed_attributes |= LightControlAttributes.COLOR

    if LightControlAttributes.COLOR not in changed_attributes:
        old_attributes, new_attributes = _add_missing_attributes(
            old_attributes,
            new_attributes,
        )

    if old_attributes.get(ATTR_BRIGHTNESS) and new_attributes.get(ATTR_BRIGHTNESS):
        last_brightness = old_attributes[ATTR_BRIGHTNESS]
        current_brightness = new_attributes[ATTR_BRIGHTNESS]
        if abs(current_brightness - last_brightness) > BRIGHTNESS_CHANGE:
            _LOGGER.debug(
                "Brightness of '%s' significantly changed from %s to %s with"
                " context.id='%s'",
                light,
                last_brightness,
                current_brightness,
                context.id,
            )
            changed_attributes |= LightControlAttributes.BRIGHTNESS

    if (
        LightControlAttributes.COLOR not in changed_attributes
        and old_attributes.get(ATTR_COLOR_TEMP_KELVIN)
        and new_attributes.get(ATTR_COLOR_TEMP_KELVIN)
    ):
        last_color_temp = old_attributes[ATTR_COLOR_TEMP_KELVIN]
        current_color_temp = new_attributes[ATTR_COLOR_TEMP_KELVIN]
        if abs(current_color_temp - last_color_temp) > COLOR_TEMP_CHANGE:
            _LOGGER.debug(
                "Color temperature of '%s' significantly changed from %s to %s with"
                " context.id='%s'",
                light,
                last_color_temp,
                current_color_temp,
                context.id,
            )
            changed_attributes |= LightControlAttributes.COLOR

    if (
        LightControlAttributes.COLOR not in changed_attributes
        and old_attributes.get(ATTR_RGB_COLOR)
        and new_attributes.get(ATTR_RGB_COLOR)
    ):
        last_rgb_color = old_attributes[ATTR_RGB_COLOR]
        current_rgb_color = new_attributes[ATTR_RGB_COLOR]
        redmean_change = color_difference_redmean(last_rgb_color, current_rgb_color)
        if redmean_change > RGB_REDMEAN_CHANGE:
            _LOGGER.debug(
                "color RGB of '%s' significantly changed from %s to %s with"
                " context.id='%s'",
                light,
                last_rgb_color,
                current_rgb_color,
                context.id,
            )
            changed_attributes |= LightControlAttributes.COLOR

    return changed_attributes


class AdaptiveSwitch(SwitchEntity, RestoreEntity):
    """Representation of a Adaptive Lighting switch."""

    def __init__(
        self,
        hass: HomeAssistant,
        config_entry: ConfigEntry,
        manager: AdaptiveLightingManager,
        sleep_mode_switch: SimpleSwitch,
        adapt_color_switch: SimpleSwitch,
        adapt_brightness_switch: SimpleSwitch,
    ) -> None:
        """Initialize the Adaptive Lighting switch."""
        # Set attributes that can't be modified during runtime
        assert hass is not None
        self.hass = hass
        self.manager = manager
        self.sleep_mode_switch = sleep_mode_switch
        self.adapt_color_switch = adapt_color_switch
        self.adapt_brightness_switch = adapt_brightness_switch

        data = validate(config_entry)

        self._name = data[CONF_NAME]
        self._interval: timedelta = data[CONF_INTERVAL]
        self.lights: list[str] = data[CONF_LIGHTS]
        self._behavior_configured_lights = frozenset(self.lights)

        # backup data for use in change_switch_settings "configuration" CONF_USE_DEFAULTS
        self._config_backup = deepcopy(data)
        self._set_changeable_settings(data=data, defaults=None)

        # Set other attributes
        self._icon = ICON_MAIN
        self._state: bool | None = None

        # To count the number of `Context` instances
        self._context_cnt: int = 0

        # Set in self._update_attrs_and_maybe_adapt_lights
        self._settings: dict[str, Any] = {}
        self._intelligence_decisions: dict[str, dict[str, Any]] = {}
        self._discovery_snapshot: InventorySnapshot | None = None
        self._discovered_context_entities: dict[str, list[str]] = {}
        self._behavior_runtime: BehaviorRuntimeAdapter | None = None
        self._behavior_runtime_started = False
        self._behavior_evaluate_task: asyncio.Task[None] | None = None
        self._behavior_abort_cleanup_task: asyncio.Task[None] | None = None
        self._intelligence_teardown_started = False
        configured_context_ids = [
            entity_id
            for value in self._intelligence_entities.values()
            for entity_id in (value if isinstance(value, list) else [value])
            if entity_id
        ]
        self._discovery = None
        if self._intelligence_enabled:
            discovery_kwargs = {
                "seed_entity_ids": [*self.lights, *configured_context_ids],
                "explicit_controlled_entity_ids": self.lights,
                "include_all_areas": True,
                "on_change": self._async_discovery_changed,
            }
            self._discovery = EntityDiscoveryCoordinator(hass, **discovery_kwargs)
        training_config = self._intelligence_training_config
        self._training = (
            AdaptiveLightingTraining(
                hass,
                storage_key=f"adaptive_lighting_training_{slugify(self._name)}",
                training_duration_days=training_config["training_days"],
                auto_promote=training_config["auto_promote"],
                minimum_samples=training_config["minimum_samples"],
                minimum_confidence=training_config["minimum_confidence"],
                durability_seconds=training_config["durability_seconds"],
                public_holiday_predicate=self._is_public_holiday,
                on_change=self._async_training_changed,
            )
            if self._intelligence_enabled and training_config["enabled"]
            else None
        )
        if self._intelligence_enabled and self._training is not None:
            self._behavior_runtime = BehaviorRuntimeAdapter(
                hass,
                storage_key=(
                    f"adaptive_lighting_behavior_runtime_{slugify(self._name)}"
                ),
                candidate_provider=self._behavior_candidates,
                context_provider=self._behavior_context,
                actuation_enabled=self._behavior_actuation_enabled,
                phase_provider=self._behavior_phase,
                on_change=self._async_behavior_changed,
                on_accepted_observation=self._async_behavior_observation_accepted,
            )

        # Set and unset tracker in async_turn_on and async_turn_off
        self.remove_listeners: list[CALLBACK_TYPE] = []
        self.remove_interval: CALLBACK_TYPE = lambda: None
        self._remove_intelligence_context_listener: CALLBACK_TYPE = lambda: None
        self._intelligence_context_refresh_task: asyncio.Task[None] | None = None
        _LOGGER.debug(
            "%s: Setting up with '%s',"
            " config_entry.data: '%s',"
            " config_entry.options: '%s', converted to '%s'.",
            self._name,
            self.lights,
            config_entry.data,
            config_entry.options,
            data,
        )

    def _set_changeable_settings(
        self,
        data: dict[str, Any],
        defaults: dict[str, Any] | None = None,
    ) -> None:
        # Only pass settings users can change during runtime
        data = validate(
            config_entry=None,
            service_data=data,
            defaults=defaults,
        )

        # backup data for use in change_switch_settings "current" CONF_USE_DEFAULTS
        self._current_settings = data

        self._detect_non_ha_changes = data[CONF_DETECT_NON_HA_CHANGES]
        self._include_config_in_attributes = data[CONF_INCLUDE_CONFIG_IN_ATTRIBUTES]
        self._config: dict[str, Any] = {}
        if self._include_config_in_attributes:
            attrdata = deepcopy(data)
            for k, v in attrdata.items():
                if isinstance(v, datetime.date | datetime.datetime):
                    attrdata[k] = v.isoformat()
                elif isinstance(v, datetime.timedelta):
                    attrdata[k] = v.total_seconds()
            self._config.update(attrdata)

        self._intelligence_enabled = data[CONF_INTELLIGENCE_ENABLED]
        self._intelligence_shadow_mode = data[CONF_INTELLIGENCE_SHADOW_MODE]
        self._intelligence_shadow_baseline_brightness = data[
            CONF_INTELLIGENCE_SHADOW_BASELINE_BRIGHTNESS
        ]
        self._intelligence_training_config = {
            "enabled": data[CONF_INTELLIGENCE_TRAINING_ENABLED],
            "training_days": data[CONF_INTELLIGENCE_TRAINING_DAYS],
            "auto_promote": data[CONF_INTELLIGENCE_AUTO_PROMOTE],
            "minimum_samples": data[CONF_INTELLIGENCE_MINIMUM_SAMPLES],
            "minimum_confidence": data[CONF_INTELLIGENCE_MINIMUM_CONFIDENCE],
            "durability_seconds": data[CONF_INTELLIGENCE_DURABILITY_SECONDS],
        }
        self._intelligence_entities = {
            CONF_OCCUPANCY_ENTITIES: list(data[CONF_OCCUPANCY_ENTITIES]),
            CONF_PRESENCE_ENTITIES: list(data[CONF_PRESENCE_ENTITIES]),
            CONF_ILLUMINANCE_ENTITIES: list(data[CONF_ILLUMINANCE_ENTITIES]),
            CONF_HOME_STATE_ENTITY: data[CONF_HOME_STATE_ENTITY],
            CONF_SECURITY_STATE_ENTITY: data[CONF_SECURITY_STATE_ENTITY],
            CONF_SLEEP_ENTITY: data[CONF_SLEEP_ENTITY],
            CONF_MEDIA_ENTITIES: list(data[CONF_MEDIA_ENTITIES]),
            CONF_ENERGY_CONSTRAINT_ENTITY: data[CONF_ENERGY_CONSTRAINT_ENTITY],
            CONF_MANUAL_HOLD_ENTITY: data[CONF_MANUAL_HOLD_ENTITY],
            CONF_SEMANTIC_INTENT_ENTITY: data[CONF_SEMANTIC_INTENT_ENTITY],
            CONF_AMBIENT_BRIGHTNESS_ENTITY: data[CONF_AMBIENT_BRIGHTNESS_ENTITY],
        }
        self._intelligence_caps = {
            "task": data[CONF_TASK_BRIGHTNESS_CAP],
            "ambient": data[CONF_AMBIENT_BRIGHTNESS_CAP],
            "video": data[CONF_VIDEO_BRIGHTNESS_CAP],
            "night": data[CONF_NIGHT_BRIGHTNESS_CAP],
            "prelight": data[CONF_PRELIGHT_BRIGHTNESS_CAP],
        }
        self._minimum_brightness = data[CONF_MIN_BRIGHTNESS]

        self.initial_transition = data[CONF_INITIAL_TRANSITION]
        self._sleep_transition = data[CONF_SLEEP_TRANSITION]
        self._only_once = data[CONF_ONLY_ONCE]
        self._prefer_rgb_color = data[CONF_PREFER_RGB_COLOR]
        self._separate_turn_on_commands = data[CONF_SEPARATE_TURN_ON_COMMANDS]
        self._transition: int = data[CONF_TRANSITION]
        self._adapt_delay = data[CONF_ADAPT_DELAY]
        self._send_split_delay = data[CONF_SEND_SPLIT_DELAY]
        self._take_over_control = data[CONF_TAKE_OVER_CONTROL]
        if not data[CONF_TAKE_OVER_CONTROL] and (
            data[CONF_DETECT_NON_HA_CHANGES] or data[CONF_ADAPT_ONLY_ON_BARE_TURN_ON]
        ):
            _LOGGER.warning(
                "%s: Config mismatch: `detect_non_ha_changes` or `adapt_only_on_bare_turn_on` "
                "set to `true` requires `take_over_control` to be enabled. Adjusting config "
                "and continuing setup with `take_over_control: true`.",
                self._name,
            )
            self._take_over_control = True
        self._take_over_control_mode = TakeOverControlMode(
            data[CONF_TAKE_OVER_CONTROL_MODE],
        )
        self._detect_non_ha_changes = data[CONF_DETECT_NON_HA_CHANGES]
        self._adapt_only_on_bare_turn_on = data[CONF_ADAPT_ONLY_ON_BARE_TURN_ON]
        self._auto_reset_manual_control_time = data[CONF_AUTORESET_CONTROL]
        self._skip_redundant_commands = data[CONF_SKIP_REDUNDANT_COMMANDS]
        self._intercept = data[CONF_INTERCEPT]
        self._multi_light_intercept = data[CONF_MULTI_LIGHT_INTERCEPT]
        if not data[CONF_INTERCEPT] and data[CONF_MULTI_LIGHT_INTERCEPT]:
            _LOGGER.warning(
                "%s: Config mismatch: `multi_light_intercept` set to `true` requires `intercept`"
                " to be enabled. Adjusting config and continuing setup with"
                " `multi_light_intercept: false`.",
                self._name,
            )
            self._multi_light_intercept = False
        self._expand_light_groups()  # updates manual control timers
        observer = get_astral_observer(self.hass)

        self._sun_light_settings = SunLightSettings(
            name=self._name,
            astral_observer=observer,
            adapt_until_sleep=data[CONF_ADAPT_UNTIL_SLEEP],
            max_brightness=data[CONF_MAX_BRIGHTNESS],
            max_color_temp=data[CONF_MAX_COLOR_TEMP],
            min_brightness=data[CONF_MIN_BRIGHTNESS],
            min_color_temp=data[CONF_MIN_COLOR_TEMP],
            sleep_brightness=data[CONF_SLEEP_BRIGHTNESS],
            sleep_color_temp=data[CONF_SLEEP_COLOR_TEMP],
            sleep_rgb_color=data[CONF_SLEEP_RGB_COLOR],
            sleep_rgb_or_color_temp=data[CONF_SLEEP_RGB_OR_COLOR_TEMP],
            sunrise_offset=data[CONF_SUNRISE_OFFSET],
            sunrise_time=data[CONF_SUNRISE_TIME],
            min_sunrise_time=data[CONF_MIN_SUNRISE_TIME],
            max_sunrise_time=data[CONF_MAX_SUNRISE_TIME],
            sunset_offset=data[CONF_SUNSET_OFFSET],
            sunset_time=data[CONF_SUNSET_TIME],
            min_sunset_time=data[CONF_MIN_SUNSET_TIME],
            max_sunset_time=data[CONF_MAX_SUNSET_TIME],
            brightness_mode=data[CONF_BRIGHTNESS_MODE],
            brightness_mode_time_dark=data[CONF_BRIGHTNESS_MODE_TIME_DARK],
            brightness_mode_time_light=data[CONF_BRIGHTNESS_MODE_TIME_LIGHT],
            timezone=zoneinfo.ZoneInfo(self.hass.config.time_zone),
        )
        _LOGGER.debug(
            "%s: Set switch settings for lights '%s'. now using data: '%s'",
            self._name,
            self.lights,
            data,
        )

    @property
    def name(self) -> str:
        """Return the name of the device if any."""
        return f"Adaptive Lighting: {self._name}"

    @property
    def unique_id(self) -> str:
        """Return the unique ID of entity."""
        return self._name

    @property
    def is_on(self) -> bool | None:
        """Return true if adaptive lighting is on."""
        return self._state

    @property
    def device_info(self) -> DeviceInfo:
        """Return the device info, used to group this and adjacent entities in the UI."""
        return DeviceInfo(
            identifiers={
                (DOMAIN, self._name),
            },
            name=self._name,
            entry_type=DeviceEntryType.SERVICE,
        )

    async def _async_discovery_changed(self, snapshot: InventorySnapshot) -> None:
        """Reconcile area/entity changes into observation-only context inputs."""
        self._discovery_snapshot = snapshot
        capability_map = {
            CONF_OCCUPANCY_ENTITIES: {"motion", "occupancy"},
            CONF_PRESENCE_ENTITIES: {"presence", "household_presence"},
            "household_presence": {"household_presence"},
            CONF_ILLUMINANCE_ENTITIES: {"illuminance"},
            CONF_MEDIA_ENTITIES: {"media"},
            "openings": {"opening", "door", "window", "garage_door"},
            "weather": {"weather"},
            "daylight": {"sun", "illuminance"},
            "solar": {"solar_proxy"},
            "holidays": {"holiday_calendar"},
            "arrivals": {"arrival"},
            "security": {"security"},
        }
        self._discovered_context_entities = {
            name: sorted(
                item.entity_id
                for item in snapshot.entities
                if item.available and capabilities.intersection(item.capabilities)
            )
            for name, capabilities in capability_map.items()
        }
        self._schedule_behavior_evaluation()
        if self.is_on:
            self._setup_intelligence_context_listener()
        if (
            self.hass.is_running
            and self.entity_id
            and self.hass.states.get(self.entity_id)
        ):
            self.async_write_ha_state()

    @staticmethod
    def _behavior_state_available(state: State | None) -> bool:
        """Return whether a live HA state can be used by behavior runtime."""
        return state is not None and state.state not in {"unknown", "unavailable"}

    def _behavior_candidates(  # noqa: PLR0912, PLR0915
        self,
    ) -> tuple[CandidateRecord, ...]:
        """Return one bounded, explicitly permitted candidate per fixture."""
        snapshot = self._discovery_snapshot
        registry = entity_registry.async_get(self.hass)
        eligible: dict[str, tuple[CandidateRecord, str | None, str | None]] = {}
        selected_entity_ids: set[str] = set()

        def is_integration_light_group(entity_id: str) -> bool:
            entry = registry.async_get(entity_id)
            state = self.hass.states.get(entity_id)
            members = state.attributes.get(ATTR_ENTITY_ID) if state else None
            return bool(
                (entry is not None and entry.platform == "group")
                or (isinstance(members, (list, tuple, set, frozenset)) and members),
            )

        def add_candidate(
            item: Any,
            *,
            device_id: str | None = None,
            fallback: bool = False,
        ) -> None:
            entity_id = getattr(item, "entity_id", None)
            domain = getattr(item, "domain", None)
            capabilities = set(getattr(item, "capabilities", ()))
            available = bool(getattr(item, "available", False))
            if fallback:
                state = self.hass.states.get(entity_id)
                if not self._behavior_state_available(state):
                    return
                available = True
            if not isinstance(entity_id, str) or domain not in {
                LIGHT_DOMAIN,
                SWITCH_DOMAIN,
            }:
                return
            is_native_light = domain == LIGHT_DOMAIN and "on_off_light" in capabilities
            is_light_switch = (
                domain == SWITCH_DOMAIN and "light_like_switch" in capabilities
            )
            if not is_native_light and not is_light_switch:
                return
            if is_native_light and is_integration_light_group(entity_id):
                return
            try:
                manual_hold = bool(
                    self.manager.get_manual_control_attributes(entity_id).has_any(),
                )
            except Exception:  # noqa: BLE001 - fail closed for discovered entities
                # A discovered entity may disappear while registry callbacks
                # are reconciling. Keep it learnable, but fail closed in the
                # runtime's per-entity actuation gate.
                _LOGGER.warning(
                    "%s: Could not read manual-control state for %s; "
                    "behavior actuation will remain blocked",
                    self._name,
                    entity_id,
                    exc_info=True,
                )
                manual_hold = True
            candidate = CandidateRecord(
                entity_id=entity_id,
                area=(
                    getattr(item, "area_name", None)
                    or getattr(item, "area_id", None)
                    or "unknown"
                ),
                domain=domain,
                supports_brightness="dimmable_light" in capabilities,
                available=available,
                explicit_light_switch=is_light_switch,
                manual_hold=manual_hold,
            )
            entry = registry.async_get(entity_id)
            state = self.hass.states.get(entity_id)
            raw_name = next(
                (
                    value
                    for value in (
                        state.attributes.get("friendly_name") if state else None,
                        getattr(entry, "name", None),
                        getattr(entry, "original_name", None),
                        getattr(item, "name", None),
                    )
                    if isinstance(value, str) and value.strip()
                ),
                None,
            )
            normalized_name = (
                " ".join(
                    raw_name.casefold().replace("_", " ").replace("-", " ").split(),
                )
                if raw_name is not None
                else None
            )
            dedupe_id = device_id or getattr(item, "device_id", None)
            eligible[entity_id] = (candidate, dedupe_id, normalized_name)
            selected_entity_ids.add(entity_id)

        if snapshot is not None:
            for item in sorted(snapshot.entities, key=lambda value: value.entity_id):
                add_candidate(item)

        snapshot_by_id = (
            {item.entity_id: item for item in snapshot.entities}
            if snapshot is not None
            else {}
        )
        fallback_ids = dict.fromkeys(
            (*self.lights, *self._behavior_configured_lights),
        )
        for entity_id in fallback_ids:
            if entity_id in selected_entity_ids:
                continue
            entry = registry.async_get(entity_id)
            state = self.hass.states.get(entity_id)
            if entry is None or not self._behavior_state_available(state):
                continue
            item = snapshot_by_id.get(entity_id)
            if item is not None:
                add_candidate(item, device_id=entry.device_id, fallback=True)
                continue
            if entry.domain != LIGHT_DOMAIN:
                # A switch is only admitted when discovery has explicitly
                # classified it as a light-like load.
                continue
            add_candidate(
                type(
                    "_ConfiguredLight",
                    (),
                    {
                        "entity_id": entity_id,
                        "domain": entry.domain,
                        "area_id": entry.area_id,
                        "area_name": None,
                        "capabilities": ("on_off_light",),
                        "available": True,
                        "device_id": entry.device_id,
                    },
                )(),
                device_id=entry.device_id,
                fallback=True,
            )

        def candidate_rank(candidate: CandidateRecord) -> tuple[int, int, str]:
            return (
                0 if candidate.available else 1,
                0 if candidate.domain == LIGHT_DOMAIN else 1,
                candidate.entity_id,
            )

        # Discovery intentionally supplies safely classified whole-house fixtures,
        # including unconfigured ones. Runtime readiness, phase, switch state, and
        # per-entity manual-hold gates decide whether a candidate may be called.
        device_selected: dict[
            tuple[str, str],
            tuple[CandidateRecord, str | None, str | None],
        ] = {}
        for candidate, device_id, normalized_name in sorted(
            eligible.values(),
            key=lambda value: value[0].entity_id,
        ):
            key = (
                ("device", device_id) if device_id else ("entity", candidate.entity_id)
            )
            previous = device_selected.get(key)
            if previous is None or candidate_rank(candidate) < candidate_rank(
                previous[0],
            ):
                device_selected[key] = (candidate, device_id, normalized_name)

        alias_selected: dict[
            tuple[str, str, str] | tuple[str, str],
            tuple[CandidateRecord, str | None, str | None],
        ] = {}
        for candidate, device_id, normalized_name in device_selected.values():
            object_id = candidate.entity_id.split(".", 1)[1]
            alias_key: tuple[str, str, str] | tuple[str, str]
            if normalized_name is not None:
                alias_key = (object_id, normalized_name, candidate.area)
            else:
                alias_key = (candidate.domain, candidate.entity_id)
            previous = alias_selected.get(alias_key)
            if previous is None or candidate_rank(candidate) < candidate_rank(
                previous[0],
            ):
                alias_selected[alias_key] = (candidate, device_id, normalized_name)

        return tuple(
            candidate
            for candidate, _, _ in sorted(
                alias_selected.values(),
                key=lambda value: value[0].entity_id,
            )
        )

    @staticmethod
    def _behavior_timestamp(
        value: datetime.datetime | None,
    ) -> datetime.datetime | None:
        """Normalize HA timestamps for the temporal runtime."""
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            return value.replace(tzinfo=dt_util.DEFAULT_TIME_ZONE).astimezone(
                dt_util.UTC,
            )
        return value.astimezone(dt_util.UTC)

    @staticmethod
    def _behavior_number(value: Any) -> float | None:
        """Parse one finite numeric sensor value without retaining raw data."""
        if isinstance(value, bool):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    @staticmethod
    def _behavior_attributes(state: State | None) -> dict[str, Any]:
        """Copy only classifier-supported scalar attributes from one HA state."""
        if state is None:
            return {}
        keys = {
            "alarm_state",
            "alarm_status",
            "app_id",
            "app_name",
            "arrived",
            "arrival",
            "arrival_signal",
            "cloud_coverage",
            "cloudiness",
            "condition",
            "content_type",
            "device_class",
            "elevation",
            "energy",
            "energy_power",
            "friendly_name",
            "illuminance",
            "irradiance",
            "media_content_type",
            "media_type",
            "media_title",
            "pv_power",
            "security_state",
            "security_status",
            "sleep",
            "sleep_mode",
            "sleeping",
            "solar_irradiance",
            "solar_power",
            "source_app",
            "sun_elevation",
            "temperature",
            "title",
            "unit_of_measurement",
            "weather_state",
        }
        return {
            key: value
            for key, value in state.attributes.items()
            if key in keys
            and (
                isinstance(value, (str, int, float, bool, datetime.datetime))
                or value is None
            )
        }

    def _behavior_solar_signal(self, state: State) -> tuple[int, float] | None:
        """Return a ranked instantaneous solar/PV signal, or fail closed."""
        attributes = self._behavior_attributes(state)
        entry = entity_registry.async_get(self.hass).async_get(state.entity_id)
        labels = (
            state.entity_id,
            attributes.get("friendly_name"),
            getattr(entry, "name", None),
            getattr(entry, "original_name", None),
            getattr(entry, "original_name_unprefixed", None),
        )
        text = " ".join(str(value) for value in labels if value).casefold()
        for separator in ("_", "-", ".", "/"):
            text = text.replace(separator, " ")
        words = set(text.split())
        device_class = str(
            attributes.get("device_class") or getattr(entry, "device_class", "") or "",
        ).casefold()
        unit = (
            str(
                attributes.get("unit_of_measurement")
                or getattr(entry, "unit_of_measurement", "")
                or "",
            )
            .strip()
            .casefold()
        )
        excluded_words = {
            "battery",
            "configuration",
            "current",
            "daily",
            "energy",
            "lifetime",
            "setting",
            "settings",
            "soc",
            "temperature",
            "today",
            "total",
            "voltage",
            "yield",
        }
        excluded_device_classes = {
            "battery",
            "current",
            "energy",
            "temperature",
            "voltage",
        }
        excluded_units = {
            "%",
            "a",
            "ah",
            "c",
            "°c",
            "hz",
            "ka",
            "kwh",
            "ma",
            "mwh",
            "v",
            "va",
            "var",
            "wh",
        }
        normalized_device_class = device_class.rsplit(".", 1)[-1]
        if (
            words.intersection(excluded_words)
            or normalized_device_class in excluded_device_classes
            or unit in excluded_units
            or unit.endswith("wh")
        ):
            return None

        attribute_signal = next(
            (
                (key, value)
                for key in (
                    "pv_power",
                    "solar_power",
                    "solar_irradiance",
                    "irradiance",
                )
                if (value := self._behavior_number(attributes.get(key))) is not None
            ),
            None,
        )
        exact_preferred = state.entity_id == "sensor.solar_inverter_pv_power"
        irradiance_semantics = "irradiance" in words
        pv_power_semantics = "power" in words and bool(
            words.intersection({"photovoltaic", "pv", "solar"}),
        )
        if (
            not exact_preferred
            and attribute_signal is None
            and not irradiance_semantics
            and not pv_power_semantics
        ):
            return None
        value = (
            attribute_signal[1]
            if attribute_signal is not None
            else self._behavior_number(state.state)
        )
        if value is None:
            return None
        if unit == "kw":
            value *= 1_000
        elif unit == "mw":
            value *= 1_000_000
        rank = (
            0
            if exact_preferred
            else 1
            if attribute_signal is not None
            and attribute_signal[0] in {"pv_power", "solar_power"}
            else 2
            if attribute_signal is not None
            else 3
            if "pv" in words and "power" in words
            else 4
            if pv_power_semantics
            else 5
        )
        return rank, value

    def _behavior_area_context(
        self,
        now: datetime.datetime,
    ) -> dict[str, dict[str, Any]]:
        """Build bounded room-local context for the runtime's area merge."""
        snapshot = self._discovery_snapshot
        if snapshot is None:
            return {}
        capability_groups = {
            "motion": {"motion"},
            "occupancy": {"occupancy"},
            "presence": {"presence", "household_presence"},
            "opening": {"opening", "door", "window", "garage_door"},
        }
        area_entities: dict[str, dict[str, list[str]]] = {}
        for item in snapshot.entities:
            area = item.area_name or item.area_id
            if not area:
                continue
            for group, capabilities in capability_groups.items():
                if capabilities.intersection(item.capabilities):
                    area_entities.setdefault(area, {}).setdefault(group, []).append(
                        item.entity_id,
                    )

        def timestamp(state: State | None) -> datetime.datetime | None:
            return self._behavior_timestamp(state.last_changed) if state else None

        def latest(states: list[State]) -> State | None:
            return max(
                states,
                key=lambda state: (
                    timestamp(state)
                    or datetime.datetime.min.replace(tzinfo=dt_util.UTC)
                ),
                default=None,
            )

        contexts: dict[str, dict[str, Any]] = {}
        for area in sorted(area_entities)[:128]:
            states_by_group = {
                group: [
                    state
                    for entity_id in entity_ids
                    if self._behavior_state_available(
                        state := self.hass.states.get(entity_id),
                    )
                ]
                for group, entity_ids in area_entities[area].items()
            }
            occupancy_states = states_by_group.get("occupancy", [])
            presence_states = states_by_group.get("presence", [])
            presence = (
                "unknown"
                if not presence_states
                else "present"
                if any(_state_is_active(state) for state in presence_states)
                else "absent"
            )
            occupancy = (
                "unknown"
                if not (occupancy_states or presence_states)
                else "occupied"
                if any(
                    _state_is_active(state)
                    for state in (*occupancy_states, *presence_states)
                )
                else "empty"
            )
            event_times: dict[str, datetime.datetime] = {}
            state_dwell: dict[str, float] = {}
            current_states: dict[str, str] = {}
            for group in ("motion", "occupancy", "presence", "opening"):
                state = latest(states_by_group.get(group, []))
                changed = timestamp(state)
                if changed is None:
                    continue
                event_times[group] = changed
                state_dwell[group] = max(0.0, (now - changed).total_seconds())
                current_states[group] = state.state
            contexts[area] = {
                "occupancy": occupancy,
                "presence": presence,
                "event_times": event_times,
                "state_dwell": state_dwell,
                "categorical_context": {
                    "occupancy": occupancy,
                    "presence": presence,
                    "motion": current_states.get("motion", "unknown"),
                    "opening": current_states.get("opening", "unknown"),
                },
            }
        return contexts

    def _behavior_context(self) -> Mapping[str, Any]:  # noqa: PLR0912, PLR0915
        """Return the bounded current context consumed by temporal behavior ML."""
        now = dt_util.utcnow()
        if now.tzinfo is None or now.utcoffset() is None:
            now = now.replace(tzinfo=dt_util.UTC)
        else:
            now = now.astimezone(dt_util.UTC)

        groups = {
            "occupancy": self._intelligence_entity_ids(CONF_OCCUPANCY_ENTITIES),
            "presence": self._intelligence_entity_ids(CONF_PRESENCE_ENTITIES),
            "arrival": self._intelligence_entity_ids("arrivals"),
            "opening": self._intelligence_entity_ids("openings"),
            "media": self._intelligence_entity_ids(CONF_MEDIA_ENTITIES),
            "security": self._intelligence_entity_ids("security"),
            "weather": self._intelligence_entity_ids("weather"),
            "daylight": self._intelligence_entity_ids("daylight"),
            "solar": self._intelligence_entity_ids("solar"),
            "holiday": self._intelligence_entity_ids("holidays"),
            "illuminance": self._intelligence_entity_ids(CONF_ILLUMINANCE_ENTITIES),
            "presence_household": self._intelligence_entity_ids(
                CONF_PRESENCE_ENTITIES,
            ),
            "household_presence": list(
                self._discovered_context_entities.get("household_presence", []),
            ),
        }
        configured_home = self._intelligence_entities.get(CONF_HOME_STATE_ENTITY)
        if configured_home:
            groups["home"] = [configured_home]
        configured_people = [
            entity_id
            for entity_id in self._intelligence_entities.get(
                CONF_PRESENCE_ENTITIES,
                [],
            )
            if entity_id.split(".", 1)[0] == "person"
        ]
        groups["household_presence"] = list(
            dict.fromkeys((*groups["household_presence"], *configured_people)),
        )
        single_ids = {
            "security": self._intelligence_single_entity(
                CONF_SECURITY_STATE_ENTITY,
                "security",
            ),
            "sleep": self._intelligence_single_entity(CONF_SLEEP_ENTITY, "sleep"),
            "energy": self._intelligence_single_entity(
                CONF_ENERGY_CONSTRAINT_ENTITY,
                "weather",
            ),
            "manual_hold": self._intelligence_single_entity(
                CONF_MANUAL_HOLD_ENTITY,
                "manual_hold",
            ),
            "semantic": self._intelligence_single_entity(
                CONF_SEMANTIC_INTENT_ENTITY,
                "semantic_intent",
            ),
        }
        for name, entity_id in single_ids.items():
            if entity_id:
                groups[name] = [entity_id]

        state_by_id = {
            entity_id: self.hass.states.get(entity_id)
            for entity_ids in groups.values()
            for entity_id in entity_ids
            if entity_id
        }

        def live_states(name: str) -> list[State]:
            return [
                state
                for entity_id in groups.get(name, [])
                if self._behavior_state_available(state := state_by_id.get(entity_id))
            ]

        def latest(name: str) -> State | None:
            return max(
                live_states(name),
                key=lambda state: (
                    self._behavior_timestamp(state.last_changed)
                    or datetime.datetime.min.replace(tzinfo=dt_util.UTC)
                ),
                default=None,
            )

        def age_seconds(state: State | None) -> float | None:
            changed = self._behavior_timestamp(state.last_changed) if state else None
            if changed is None:
                return None
            return max(0.0, (now - changed).total_seconds())

        occupancy_states = live_states("occupancy")
        presence_states = live_states("presence")
        home_state = latest("home")
        home_classification = classify_household_state(
            {
                "state": home_state.state if home_state else None,
                "attributes": self._behavior_attributes(home_state),
                "entity_id": home_state.entity_id if home_state else None,
            },
        )
        presence_value = (
            "unknown"
            if not presence_states
            else "present"
            if any(_state_is_active(state) for state in presence_states)
            else "absent"
        )
        occupancy_value = (
            "unknown"
            if not (occupancy_states or presence_states)
            else "occupied"
            if any(
                _state_is_active(state)
                for state in (*occupancy_states, *presence_states)
            )
            else "empty"
        )
        home_away = home_classification.category.value
        if home_away == "unknown":
            household_classifications = [
                classify_household_state(
                    {
                        "state": state.state,
                        "attributes": self._behavior_attributes(state),
                        "entity_id": state.entity_id,
                    },
                ).category.value
                for state in live_states("household_presence")
            ]
            if "home" in household_classifications:
                home_away = "home"
            elif household_classifications and all(
                value == "away" for value in household_classifications
            ):
                home_away = "away"

        arrival_state = latest("arrival")
        arrival_age = age_seconds(arrival_state)
        arrival_classification = classify_arrival_state(
            {
                "state": arrival_state.state if arrival_state else None,
                "attributes": self._behavior_attributes(arrival_state),
                "entity_id": arrival_state.entity_id if arrival_state else None,
            },
            observed_at=(
                self._behavior_timestamp(arrival_state.last_changed)
                if arrival_state
                else None
            ),
            age_seconds=arrival_age,
            max_age_seconds=15 * 60,
            now=now,
        )
        recent_arrival = arrival_classification.category is ArrivalState.RECENT

        media_states = live_states("media")
        active_media = [
            state
            for state in media_states
            if state.state.lower() in {"playing", "paused", "buffering", "on"}
        ]
        media_state = max(
            active_media,
            key=lambda state: (
                self._behavior_timestamp(state.last_changed)
                or datetime.datetime.min.replace(tzinfo=dt_util.UTC)
            ),
            default=None,
        ) or latest("media")
        media_attributes = self._behavior_attributes(media_state)
        media_classification = classify_media_state(
            {
                "state": media_state.state if media_state else None,
                "attributes": media_attributes,
                "entity_id": media_state.entity_id if media_state else None,
            },
        )
        media_category = media_classification.category.value

        security_state = latest("security")
        security_attributes = self._behavior_attributes(security_state)
        security_classification = classify_security_state(
            {
                "state": security_state.state if security_state else None,
                "attributes": security_attributes,
                "entity_id": security_state.entity_id if security_state else None,
            },
        )
        security_category = security_classification.category.value
        emergency = security_classification.category is SecurityState.EMERGENCY
        safety = emergency or security_category == "problem"

        daylight_states = [*live_states("daylight"), *live_states("illuminance")]
        solar_states = live_states("solar")
        weather_state = latest("weather")
        daylight_state = max(
            daylight_states,
            key=lambda state: (
                self._behavior_timestamp(state.last_changed)
                or datetime.datetime.min.replace(tzinfo=dt_util.UTC)
            ),
            default=None,
        )
        weather_attributes = self._behavior_attributes(weather_state)
        weather_classification = classify_weather_daylight(
            {
                "state": weather_state.state if weather_state else None,
                "attributes": weather_attributes,
                "entity_id": weather_state.entity_id if weather_state else None,
            },
        )
        weather_category = weather_classification.category.value
        weather_condition = (
            weather_attributes.get("condition")
            or weather_attributes.get("weather_state")
            or (weather_state.state if weather_state else None)
        )

        daylight_attributes = self._behavior_attributes(daylight_state)
        daylight_classification = classify_weather_daylight(
            {
                "state": daylight_state.state if daylight_state else None,
                "attributes": daylight_attributes,
                "entity_id": daylight_state.entity_id if daylight_state else None,
            },
        )
        illuminance_state = next(
            (
                state
                for state in live_states("illuminance")
                if self._behavior_number(state.state) is not None
            ),
            None,
        )
        illuminance = self._behavior_number(
            illuminance_state.state if illuminance_state else None,
        )
        energy_state = latest("energy")
        energy_value = self._behavior_number(
            energy_state.state if energy_state else None,
        )
        ranked_solar: list[tuple[int, float, float]] = []
        for state in solar_states:
            signal = self._behavior_solar_signal(state)
            if signal is None:
                continue
            rank, value = signal
            changed = (
                self._behavior_timestamp(state.last_changed)
                or datetime.datetime.min.replace(tzinfo=dt_util.UTC)
            ).timestamp()
            ranked_solar.append((rank, -changed, value))
        solar_value = min(ranked_solar)[2] if ranked_solar else None
        daylight_band = daylight_classification.category.value
        if illuminance is not None:
            daylight_band = (
                "dark" if illuminance < 30 else "dim" if illuminance < 150 else "bright"
            )
        solar_band = (
            "unknown"
            if solar_value is None
            else "none"
            if solar_value <= 0
            else "low"
            if solar_value < 120
            else "high"
        )
        energy_band = (
            "unknown"
            if energy_value is None
            else "low"
            if energy_value <= 0
            else "high"
        )

        semantic_state = latest("semantic")
        semantic_value = semantic_state.state if semantic_state else ""
        if not isinstance(semantic_value, str):
            semantic_value = ""
        semantic_value = (
            semantic_value.strip().lower().replace("-", "_").replace(" ", "_")
        )
        sleep_state = latest("sleep")
        sleep = _state_is_active(sleep_state) or any(
            token in semantic_value for token in ("sleep", "sleeping", "good_night")
        )
        good_night = (
            sleep or "good_night" in semantic_value or semantic_value == "goodnight"
        )
        semantic_routine = "good_night" if good_night else semantic_value or "ambient"
        manual_hold_state = latest("manual_hold")
        manual_hold = _state_is_active(manual_hold_state)

        holiday_state = latest("holiday")
        holiday = _state_is_active(holiday_state)
        holiday_provenance = None
        if holiday_state is not None:
            holiday_provenance = (
                self._behavior_attributes(holiday_state).get("friendly_name")
                or holiday_state.entity_id
            )
        local_now = dt_util.as_local(now)
        day_type = "weekend" if local_now.weekday() >= 5 else "weekday"

        event_times: dict[str, datetime.datetime] = {}
        state_dwell: dict[str, float] = {}
        event_groups = {
            "occupancy": "occupancy",
            "presence": "presence",
            "arrival": "arrival",
            "opening": "opening",
            "media": "media",
            "security": "security",
            "home": "home",
        }
        for event_name, group_name in event_groups.items():
            state = latest(group_name)
            changed = self._behavior_timestamp(state.last_changed) if state else None
            if changed is not None:
                event_times[event_name] = changed
                state_dwell[event_name] = max(0.0, (now - changed).total_seconds())

        categorical_context = {
            "occupancy": occupancy_value,
            "presence": presence_value,
            "home_away": home_away,
            "arrival": "recent" if recent_arrival else "inactive",
            "media": media_category,
            "security": security_category,
            "weather": weather_category,
            "daylight": daylight_band,
            "solar": solar_band,
            "energy": energy_band,
            "routine": semantic_routine,
            "mode": "sleep" if sleep else home_away,
            "day_type": day_type,
            "manual_hold": "held" if manual_hold else "free",
        }
        sun_elevation = daylight_attributes.get("sun_elevation")
        if sun_elevation is None:
            sun_elevation = daylight_attributes.get("elevation")
        return {
            "timestamp": now,
            "occupancy": occupancy_value,
            "occupancy_available": bool(occupancy_states or presence_states),
            "presence": presence_value,
            "presence_available": bool(presence_states),
            "home_away": home_away,
            "home_state": home_away,
            "recent_arrival": recent_arrival,
            "arrival_timestamp": (
                self._behavior_timestamp(arrival_state.last_changed)
                if arrival_state and arrival_state.last_changed
                else None
            ),
            "semantic_routine": semantic_routine,
            "good_night": good_night,
            "sleep": sleep,
            "media": media_category,
            "media_category": media_category,
            "media_state": media_state.state if media_state else "unknown",
            "app_name": media_attributes.get("app_name"),
            "source_app": media_attributes.get("source_app")
            or media_attributes.get("app_id"),
            "media_content_type": media_attributes.get("media_content_type")
            or media_attributes.get("media_type")
            or media_attributes.get("content_type"),
            "alarm": security_category,
            "alarm_state": security_state.state if security_state else None,
            "security": security_category,
            "safety": safety,
            "emergency": emergency,
            "weather": weather_category,
            "weather_condition": weather_condition,
            "daylight": daylight_band,
            "daylight_band": daylight_band,
            "sun": sun_elevation,
            "illuminance": illuminance,
            "solar": solar_value,
            "solar_band": solar_band,
            "energy": energy_value,
            "energy_band": energy_band,
            "is_holiday": holiday,
            "holiday": holiday,
            "holiday_name": holiday_provenance,
            "holiday_provenance": holiday_provenance,
            "is_weekend": day_type == "weekend",
            "day_type": day_type,
            "manual_hold": manual_hold,
            "categorical_context": categorical_context,
            "event_times": event_times,
            "state_dwell": state_dwell,
            "area_context": self._behavior_area_context(now),
        }

    def _behavior_phase(self) -> str:
        """Return the training phase exposed to the temporal runtime."""
        return self._training.phase if self._training is not None else "disabled"

    def _behavior_actuation_enabled(self) -> bool:
        """Gate behavior service calls behind the complete rollout contract."""
        return bool(
            self.is_on
            and self._behavior_runtime_started
            and self._training is not None
            and self._training.is_active
            and self._intelligence_training_config["auto_promote"]
            and not self._intelligence_shadow_actuation_blocked,
        )

    async def _async_behavior_changed(self, _: Mapping[str, Any]) -> None:
        """Publish bounded runtime diagnostics without breaking HA callbacks."""
        if (
            self.hass.is_running
            and self.entity_id
            and self.hass.states.get(self.entity_id)
        ):
            self.async_write_ha_state()

    async def _async_behavior_observation_accepted(self, observation: Any) -> None:
        """Count accepted on/off and Good Night behavior toward commissioning."""
        if self._training is not None:
            await self._training.async_record_behavior_observation(observation)

    def _schedule_behavior_evaluation(self) -> None:
        """Coalesce discovery/context changes into one runtime evaluation."""
        if self._behavior_runtime is None or not self._behavior_runtime_started:
            return
        if (
            self._behavior_evaluate_task is not None
            and not self._behavior_evaluate_task.done()
        ):
            return
        self._behavior_evaluate_task = self.hass.async_create_task(
            self._async_evaluate_behavior(),
            name=f"adaptive_lighting_behavior_{slugify(self._name)}",
        )

    async def _async_evaluate_behavior(self) -> None:
        """Run one behavior evaluation and always clear its tracked task."""
        task = asyncio.current_task()
        try:
            if self._behavior_runtime is not None:
                await self._behavior_runtime.async_evaluate()
        except asyncio.CancelledError:
            raise
        except Exception:
            _LOGGER.exception("Adaptive Lighting behavior evaluation failed")
        finally:
            if self._behavior_evaluate_task is task:
                self._behavior_evaluate_task = None

    def _cancel_behavior_evaluation(self) -> asyncio.Task[None] | None:
        """Synchronously block and cancel the one coalesced behavior task."""
        task = self._behavior_evaluate_task
        self._behavior_evaluate_task = None
        if task is not None and not task.done():
            task.cancel()
        return task

    async def _async_await_behavior_cancellation(
        self,
        task: asyncio.Task[None] | None,
    ) -> None:
        """Await behavior cancellation so teardown cannot race actuation."""
        if task is None or task is asyncio.current_task():
            return
        await asyncio.gather(task, return_exceptions=True)

    async def _async_start_behavior_runtime(self) -> None:
        """Start one behavior runtime only while this switch is enabled."""
        if self._behavior_runtime is None or self._behavior_runtime_started:
            return
        await self._behavior_runtime.async_start()
        self._behavior_runtime_started = True

    async def _async_stop_behavior_runtime(
        self,
        task: asyncio.Task[None] | None = None,
    ) -> None:
        """Await any evaluation before stopping all runtime listeners."""
        self._behavior_runtime_started = False
        task = task or self._cancel_behavior_evaluation()
        await self._async_await_behavior_cancellation(task)
        if self._behavior_runtime is not None:
            await self._behavior_runtime.async_stop()

    async def _async_abort_intelligence_cleanup(
        self,
        task: asyncio.Task[None] | None,
    ) -> None:
        """Finish cleanup when HA aborts entity platform addition."""
        await self._async_stop_behavior_runtime(task)
        if self._discovery is not None:
            await self._discovery.async_stop()
        if self._training is not None:
            await self._training.async_unload()

    def _clear_behavior_abort_cleanup(self, task: asyncio.Task[None]) -> None:
        """Clear only the abort task whose completion callback is running."""
        if self._behavior_abort_cleanup_task is task:
            self._behavior_abort_cleanup_task = None

    def _is_public_holiday(self, _: datetime.date) -> bool:
        """Resolve today's holiday from configured/auto-discovered local entities."""
        return any(
            _state_is_active(self.hass.states.get(entity_id))
            for entity_id in self._discovered_context_entities.get("holidays", [])
        )

    async def _async_training_changed(self, _: dict[str, Any]) -> None:
        """Publish phase/sample changes without granting a new actuation path."""
        if (
            self.hass.is_running
            and self.entity_id
            and self.hass.states.get(self.entity_id)
        ):
            self.async_write_ha_state()

    def _intelligence_entity_ids(self, name: str) -> list[str]:
        """Merge configured entities with live discovery without duplicates."""
        configured = self._intelligence_entities.get(name, [])
        configured_ids = configured if isinstance(configured, list) else [configured]
        return list(
            dict.fromkeys(
                entity_id
                for entity_id in (
                    *configured_ids,
                    *self._discovered_context_entities.get(name, []),
                )
                if entity_id
            ),
        )

    def _intelligence_single_entity(
        self,
        name: str,
        discovered_group: str,
    ) -> str | None:
        """Prefer explicit singleton context, then one available discovery result."""
        configured = self._intelligence_entities.get(name)
        if isinstance(configured, str) and configured:
            return configured
        return next(
            iter(self._discovered_context_entities.get(discovered_group, [])),
            None,
        )

    def _setup_intelligence_context_listener(self) -> None:
        """Track configured/discovered context and replace stale registry targets."""
        self._remove_intelligence_context_listener()
        if not self._intelligence_enabled:
            self._remove_intelligence_context_listener = lambda: None
            return
        entity_ids = sorted(
            {
                entity_id
                for value in self._intelligence_entities.values()
                for entity_id in (value if isinstance(value, list) else [value])
                if entity_id
            }
            | {
                entity_id
                for entity_ids in self._discovered_context_entities.values()
                for entity_id in entity_ids
            },
        )
        self._remove_intelligence_context_listener = (
            async_track_state_change_event(
                self.hass,
                entity_ids,
                self._schedule_intelligence_context_refresh,
            )
            if entity_ids
            else lambda: None
        )

    @callback
    def _schedule_intelligence_context_refresh(
        self,
        _: EventStateChangedData,
    ) -> None:
        """Coalesce noisy sensor bursts while keeping alarms responsive."""
        self._schedule_behavior_evaluation()
        if not self.is_on or (
            self._intelligence_context_refresh_task is not None
            and not self._intelligence_context_refresh_task.done()
        ):
            return
        self._intelligence_context_refresh_task = self.hass.async_create_task(
            self._async_refresh_from_context(),
            name=f"adaptive_lighting_context_{slugify(self._name)}",
        )

    async def _async_refresh_from_context(self) -> None:
        """Re-evaluate a settled context snapshot through normal safety gates."""
        try:
            await asyncio.sleep(0.25)
            if self.is_on:
                await self._update_attrs_and_maybe_adapt_lights(
                    context=self.create_context("intelligence_context"),
                    transition=self._transition,
                )
        finally:
            self._intelligence_context_refresh_task = None

    async def async_added_to_hass(self) -> None:
        """Call when entity about to be added to hass."""
        if self._discovery is not None:
            await self._discovery.async_start()
        if self._training is not None:
            await self._training.async_setup()
        if self.hass.is_running:
            await self._setup_listeners()
        else:
            self.hass.bus.async_listen_once(
                EVENT_HOMEASSISTANT_STARTED,
                self._setup_listeners,
            )
        last_state: State | None = await self.async_get_last_state()
        is_new_entry = last_state is None  # newly added to HA
        if is_new_entry or last_state.state == STATE_ON:  # type: ignore[union-attr]
            await self.async_turn_on(adapt_lights=not self._only_once)
        else:
            self._state = False
            await self._async_stop_behavior_runtime()
            assert not self.remove_listeners

    async def async_will_remove_from_hass(self) -> None:
        """Remove the listeners upon removing the component."""
        self._intelligence_teardown_started = True
        self._behavior_runtime_started = False
        behavior_task = self._cancel_behavior_evaluation()
        self._remove_listeners()
        if self._behavior_abort_cleanup_task is not None:
            await self._behavior_abort_cleanup_task
            return
        await self._async_stop_behavior_runtime(behavior_task)
        if self._discovery is not None:
            await self._discovery.async_stop()
        if self._training is not None:
            await self._training.async_unload()

    def _expand_light_groups(self, hass: HomeAssistant | None = None) -> None:
        hass = hass or self.hass
        all_lights = _expand_light_groups(hass, self.lights)
        self.manager.lights.update(all_lights)
        self.manager.set_auto_reset_manual_control_times(
            all_lights,
            self._auto_reset_manual_control_time,
        )
        self.lights = list(all_lights)

    async def _setup_listeners(self, _: Event[NoEventData] | None = None) -> None:
        _LOGGER.debug("%s: Called '_setup_listeners'", self._name)
        if not self.is_on or not self.hass.is_running:
            _LOGGER.debug("%s: Cancelled '_setup_listeners'", self._name)
            return

        assert not self.remove_listeners

        self._update_time_interval_listener()

        remove_sleep = async_track_state_change_event(
            self.hass,
            entity_ids=self.sleep_mode_switch.entity_id,
            action=self._sleep_mode_switch_state_event_action,
        )

        self.remove_listeners.append(remove_sleep)
        self._expand_light_groups()
        self._setup_intelligence_context_listener()

    def _update_time_interval_listener(self) -> None:
        """Create or recreate the adaptation interval listener.

        Recreation is necessary when the configuration has changed (e.g., `send_split_delay`).
        """
        self._remove_interval_listener()

        # An adaptation takes a little longer than its nominal duration due processing overhead,
        # so we factor this in to avoid overlapping adaptations. Since this is a constant value,
        # it might not cover all cases, but if large enough, it covers most.
        # Ideally, the interval and adaptation are a coupled process where a finished adaptation
        # triggers the next, but that requires a larger architectural change.
        processing_overhead_time = 0.5

        adaptation_interval = (
            self._interval
            + timedelta(milliseconds=self._send_split_delay)
            + timedelta(seconds=processing_overhead_time)
        )

        self.remove_interval = async_track_time_interval(
            self.hass,
            action=self._async_update_at_interval_action,
            interval=adaptation_interval,
        )

    def _call_on_remove_callbacks(self) -> None:
        """Call callbacks registered by async_on_remove."""
        # This is called when the integration is removed from HA
        # and in `Entity.add_to_platform_abort`.
        # For some unknown reason (to me) `async_will_remove_from_hass`
        # is not called in `add_to_platform_abort`.
        # See https://github.com/basnijholt/adaptive-lighting/issues/658
        self._behavior_runtime_started = False
        behavior_task = self._cancel_behavior_evaluation()
        self._remove_listeners()
        if (
            not self._intelligence_teardown_started
            and self._behavior_abort_cleanup_task is None
        ):
            self._intelligence_teardown_started = True
            cleanup_task = self.hass.async_create_task(
                self._async_abort_intelligence_cleanup(behavior_task),
                name=f"adaptive_lighting_abort_cleanup_{slugify(self._name)}",
            )
            self._behavior_abort_cleanup_task = cleanup_task
            cleanup_task.add_done_callback(self._clear_behavior_abort_cleanup)
            if cleanup_task.done():
                self._clear_behavior_abort_cleanup(cleanup_task)
        try:
            # HACK: this is a private method in `Entity` which can change
            super()._call_on_remove_callbacks()
        except AttributeError:
            _LOGGER.exception(
                "%s: Caught AttributeError in `_call_on_remove_callbacks`",
                self._name,
            )

    def _remove_interval_listener(self) -> None:
        self.remove_interval()
        self.remove_interval = lambda: None

    def _remove_listeners(self) -> None:
        self._remove_interval_listener()
        self._remove_intelligence_context_listener()
        self._remove_intelligence_context_listener = lambda: None
        if (
            self._intelligence_context_refresh_task is not None
            and not self._intelligence_context_refresh_task.done()
        ):
            self._intelligence_context_refresh_task.cancel()
        self._intelligence_context_refresh_task = None
        while self.remove_listeners:
            remove_listener = self.remove_listeners.pop()
            remove_listener()

    @property
    def _intelligence_shadow_actuation_blocked(self) -> bool:
        """Return whether intelligence shadow mode must suppress all AL calls."""
        if not self._intelligence_enabled:
            return False
        if self._training is not None and not self._training.is_active:
            return True
        auto_promoted = (
            self._training is not None
            and self._training.is_active
            and self._intelligence_training_config["auto_promote"]
        )
        return self._intelligence_shadow_mode and not auto_promoted

    @property
    def _intelligence_shadow_baseline_brightness_active(self) -> bool:
        """Return whether the opt-in deterministic shadow baseline may run.

        This is intentionally narrower than actuation permission. It may add or
        update brightness only when an external call is already turning a
        configured light on, or while that light is already on. It is never
        consulted by the learned power-state behavior runtime.
        """
        return bool(
            self._intelligence_shadow_actuation_blocked
            and self._intelligence_shadow_baseline_brightness,
        )

    def _intelligence_snapshot(self) -> dict[str, Any]:  # noqa: PLR0912, PLR0915
        """Build a serializable snapshot for the optional pure policy engine."""
        signals: list[dict[str, Any]] = []

        def add_signal(kind: str, entity_id: str | None) -> State | None:
            if not entity_id:
                return None
            state = self.hass.states.get(entity_id)
            signals.append(
                {
                    "name": kind,
                    "kind": kind,
                    "entity_id": entity_id,
                    "source": entity_id,
                    "state": state.state if state is not None else None,
                    "value": state.state if state is not None else None,
                    "confidence": 1.0 if state is not None else 0.0,
                },
            )
            return state

        def add_signals(kind: str, entity_ids: list[str]) -> list[State]:
            return [
                state
                for entity_id in entity_ids
                if (state := add_signal(kind, entity_id)) is not None
            ]

        occupancy_states = add_signals(
            CONF_OCCUPANCY_ENTITIES,
            self._intelligence_entity_ids(CONF_OCCUPANCY_ENTITIES),
        )
        presence_states = add_signals(
            CONF_PRESENCE_ENTITIES,
            self._intelligence_entity_ids(CONF_PRESENCE_ENTITIES),
        )
        illuminance_states = add_signals(
            CONF_ILLUMINANCE_ENTITIES,
            self._intelligence_entity_ids(CONF_ILLUMINANCE_ENTITIES),
        )
        media_states = add_signals(
            CONF_MEDIA_ENTITIES,
            self._intelligence_entity_ids(CONF_MEDIA_ENTITIES),
        )
        discovered_single_groups = {
            CONF_HOME_STATE_ENTITY: CONF_PRESENCE_ENTITIES,
            CONF_SECURITY_STATE_ENTITY: "security",
            CONF_SLEEP_ENTITY: "sleep",
            CONF_ENERGY_CONSTRAINT_ENTITY: "weather",
            CONF_MANUAL_HOLD_ENTITY: "manual_hold",
            CONF_SEMANTIC_INTENT_ENTITY: "semantic_intent",
            CONF_AMBIENT_BRIGHTNESS_ENTITY: "ambient_brightness",
        }
        single_states = {
            name: add_signal(
                name,
                self._intelligence_single_entity(
                    name,
                    discovered_single_groups[name],
                ),
            )
            for name in (
                CONF_HOME_STATE_ENTITY,
                CONF_SECURITY_STATE_ENTITY,
                CONF_SLEEP_ENTITY,
                CONF_ENERGY_CONSTRAINT_ENTITY,
                CONF_MANUAL_HOLD_ENTITY,
                CONF_SEMANTIC_INTENT_ENTITY,
                CONF_AMBIENT_BRIGHTNESS_ENTITY,
            )
        }
        semantic_state = single_states[CONF_SEMANTIC_INTENT_ENTITY]
        semantic_intent = (
            semantic_state.state.lower() if semantic_state is not None else None
        )
        media_playing = any(_state_is_active(state) for state in media_states)
        media_categories = [
            classify_media_state(
                {
                    "state": state.state,
                    "attributes": state.attributes,
                    "entity_id": state.entity_id,
                },
            ).category
            for state in media_states
            if _state_is_active(state)
        ]
        # Explicit helper entity names are an adapter-level contract. A binary
        # sensor named ``*_video_content_playing`` carries stronger meaning than
        # a generic media_player whose app/content metadata is absent.
        if any(
            _state_is_active(state) and "video" in state.entity_id
            for state in media_states
        ):
            media_categories.insert(0, MediaState.VIDEO)
        media_category = next(
            (
                category
                for category in (
                    MediaState.MOVIE,
                    MediaState.TV,
                    MediaState.VIDEO,
                    MediaState.GAME,
                    MediaState.PODCAST,
                    MediaState.MUSIC,
                    MediaState.AUDIO,
                )
                if category in media_categories
            ),
            MediaState.UNKNOWN,
        )
        security_state = single_states[CONF_SECURITY_STATE_ENTITY]
        security = classify_security_state(
            {
                "state": security_state.state if security_state is not None else None,
                "attributes": security_state.attributes
                if security_state is not None
                else {},
                "entity_id": security_state.entity_id
                if security_state is not None
                else None,
            },
        )
        sleep_active = _state_is_active(single_states[CONF_SLEEP_ENTITY])
        manual_hold = _state_is_active(single_states[CONF_MANUAL_HOLD_ENTITY])
        ambient_brightness_state = single_states[CONF_AMBIENT_BRIGHTNESS_ENTITY]
        try:
            ambient_brightness_value = (
                float(ambient_brightness_state.state)
                if ambient_brightness_state is not None
                else None
            )
        except (TypeError, ValueError):
            ambient_brightness_value = None

        intent_hint = "adaptive"
        semantic_value = semantic_intent or ""
        if security.category is SecurityState.EMERGENCY:
            intent_hint = "emergency"
        elif sleep_active or any(
            token in semantic_value for token in ("night", "sleep")
        ):
            intent_hint = "sleep"
        elif any(token in semantic_value for token in ("video", "movie", "cinema")):
            intent_hint = "video"
        elif any(token in semantic_value for token in ("task", "focus", "work")):
            intent_hint = "task"
        elif any(token in semantic_value for token in ("ambient", "relax", "chill")):
            intent_hint = "ambient"
        elif "prelight" in semantic_value:
            intent_hint = "prelight"
        elif media_category in {MediaState.MOVIE, MediaState.TV, MediaState.VIDEO}:
            intent_hint = "video"
        elif media_category is MediaState.GAME:
            intent_hint = "task"
        elif media_category in {
            MediaState.PODCAST,
            MediaState.MUSIC,
            MediaState.AUDIO,
        }:
            intent_hint = "ambient"

        illuminance_value: float | None = None
        for state in illuminance_states:
            try:
                illuminance_value = float(state.state)
            except ValueError:
                continue
            break

        return {
            "signals": signals,
            "occupancy": any(
                _state_is_active(state)
                for state in (*occupancy_states, *presence_states)
            ),
            "occupancy_available": bool(occupancy_states or presence_states),
            "presence": any(_state_is_active(state) for state in presence_states),
            "presence_available": bool(presence_states),
            "illuminance": [state.state for state in illuminance_states],
            "illuminance_value": illuminance_value,
            "illuminance_available": illuminance_value is not None,
            "ambient_brightness_value": ambient_brightness_value,
            "ambient_brightness_available": ambient_brightness_value is not None,
            "home_state": single_states[CONF_HOME_STATE_ENTITY].state
            if single_states[CONF_HOME_STATE_ENTITY] is not None
            else None,
            "security_state": single_states[CONF_SECURITY_STATE_ENTITY].state
            if single_states[CONF_SECURITY_STATE_ENTITY] is not None
            else None,
            "sleep": sleep_active,
            "sleep_available": single_states[CONF_SLEEP_ENTITY] is not None,
            "emergency": security.category is SecurityState.EMERGENCY,
            "emergency_available": security.is_known,
            "media_playing": media_playing,
            "video_playing": media_category
            in {MediaState.MOVIE, MediaState.TV, MediaState.VIDEO},
            "media_available": bool(media_states),
            "media_category": media_category.value,
            "energy_constraint": single_states[CONF_ENERGY_CONSTRAINT_ENTITY].state
            if single_states[CONF_ENERGY_CONSTRAINT_ENTITY] is not None
            else None,
            "manual_hold": manual_hold,
            "manual_hold_available": single_states[CONF_MANUAL_HOLD_ENTITY] is not None,
            "semantic_intent": semantic_intent,
            "semantic_intent_available": semantic_state is not None,
            "intent_hint": intent_hint,
        }

    def _fallback_intelligence_decision(
        self,
        light: str,
        baseline_brightness_pct: int,
        snapshot: dict[str, Any],
    ) -> dict[str, Any]:
        """Return a conservative policy result when the pure engine is absent."""
        light_on = is_on(self.hass, light)
        manually_controlled = self.manager.get_manual_control_attributes(
            light,
        ).has_any()
        manual_hold = snapshot["manual_hold"] or manually_controlled
        intent = snapshot["intent_hint"]
        signals = [signal["entity_id"] for signal in snapshot["signals"]]
        provenance = {
            "adapter": "runtime_fallback",
            "signals": signals,
            "manual_control": manually_controlled,
        }

        if not light_on:
            return {
                "intent": intent,
                "baseline_brightness_pct": baseline_brightness_pct,
                "target_brightness_pct": baseline_brightness_pct,
                "confidence": 1.0,
                "reason": "light_not_already_on; automatic turn-on is not implemented",
                "provenance": provenance,
                "can_adjust": False,
                "can_turn_on": False,
                "can_turn_off": False,
                "overridden": False,
            }
        if manual_hold:
            return {
                "intent": "manual_hold",
                "baseline_brightness_pct": baseline_brightness_pct,
                "target_brightness_pct": baseline_brightness_pct,
                "confidence": 1.0,
                "reason": "manual hold or existing Adaptive Lighting takeover is authoritative",
                "provenance": provenance,
                "can_adjust": False,
                "can_turn_on": False,
                "can_turn_off": False,
                "overridden": False,
            }

        return {
            "intent": intent,
            "baseline_brightness_pct": baseline_brightness_pct,
            "target_brightness_pct": baseline_brightness_pct,
            "confidence": 0.0,
            "reason": "policy unavailable; preserving deterministic baseline",
            "provenance": provenance,
            "can_adjust": False,
            "can_turn_on": False,
            "can_turn_off": False,
            "overridden": False,
        }

    def _intelligence_decision(
        self,
        light: str,
        baseline_brightness_pct: int,
        *,
        external_turn_on: bool = False,
    ) -> dict[str, Any]:
        """Evaluate one light while preserving manual and turn-on safety boundaries.

        ``external_turn_on`` means Home Assistant is already processing a
        caller-owned turn-on request. It permits selecting the brightness that
        is added to that request; it never grants this integration permission
        to originate the power-state change.
        """
        snapshot = self._intelligence_snapshot()
        snapshot["manual_control"] = self.manager.get_manual_control_attributes(
            light,
        ).has_any()
        fallback = self._fallback_intelligence_decision(
            light,
            baseline_brightness_pct,
            snapshot,
        )
        if not self._intelligence_enabled:
            return fallback

        light_state = self.hass.states.get(light)
        current_brightness = (
            round(100 * light_state.attributes[ATTR_BRIGHTNESS] / 255)
            if light_state is not None
            and isinstance(light_state.attributes.get(ATTR_BRIGHTNESS), (int, float))
            else baseline_brightness_pct
        )
        light_available_for_adjustment = is_on(self.hass, light) or external_turn_on
        policy_values = {
            **self._intelligence_caps,
            "baseline_brightness_pct": baseline_brightness_pct,
            "current_brightness_pct": current_brightness,
            "light_on": light_available_for_adjustment,
            "manual_hold": snapshot["manual_hold"],
        }
        external = _evaluate_intelligence_policy(
            snapshot,
            policy_values,
            snapshot["intent_hint"],
        )
        if not external:
            return fallback

        decision = dict(fallback)
        decision["intent"] = _json_safe(external.get("intent", fallback["intent"]))
        decision["confidence"] = float(
            external.get("confidence", fallback["confidence"]),
        )
        decision["reason"] = str(external.get("reason", fallback["reason"]))
        decision["provenance"] = _json_safe(
            external.get("provenance", fallback["provenance"]),
        )
        for permission in ("can_adjust", "can_turn_on", "can_turn_off"):
            decision[permission] = bool(external.get(permission, False))
        target = external.get(
            "target_brightness_pct",
            external.get("target_brightness", fallback["target_brightness_pct"]),
        )
        if (
            self._training is not None
            and self._training.is_active
            and decision["can_adjust"]
        ):
            time_bucket, daylight_band = self._intelligence_learning_dimensions(
                snapshot,
            )
            preference = self._training.preference_for(
                baseline=float(target),
                zone=self._name,
                intent=str(decision["intent"]),
                time_bucket=time_bucket,
                daylight_band=daylight_band,
            )
            qualified = (
                preference["samples"] >= 2
                and preference["confidence"]
                >= self._intelligence_training_config["minimum_confidence"]
            )
            decision["learning"] = {**preference, "qualified": qualified}
            if qualified:
                target = preference["target"]
                decision["reason"] += (
                    f" | learned {preference['offset']:+g}% from "
                    f"{preference['samples']} matching human samples"
                )
        try:
            # Keep active intelligence adjustment-only. Some integrations
            # interpret brightness=0 as a power-off command.
            target = max(1, min(baseline_brightness_pct, round(float(target))))
        except (TypeError, ValueError):
            target = fallback["target_brightness_pct"]
        if (
            not decision["can_adjust"]
            or not light_available_for_adjustment
            or snapshot["manual_hold"]
        ):
            if decision["intent"] not in {"manual", "manual_hold"}:
                target = baseline_brightness_pct
            decision["reason"] = (
                "policy did not authorize adjustment; preserving baseline"
                if not decision["can_adjust"]
                else "light_not_already_on; automatic turn-on is not implemented"
                if not light_available_for_adjustment
                else "manual hold is authoritative"
            )
        decision["target_brightness_pct"] = target
        decision["overridden"] = target != baseline_brightness_pct
        return decision

    def _refresh_intelligence_decisions(self) -> None:
        """Refresh exposed decisions without causing any light service call."""
        if not self._intelligence_enabled:
            self._intelligence_decisions.clear()
            return
        baseline = self._intelligence_baseline_brightness()
        self._intelligence_decisions = {
            light: self._intelligence_decision(light, baseline) for light in self.lights
        }

    def _intelligence_baseline_brightness(self) -> int:
        """Return a current sun baseline even when the restored switch is off."""
        if "brightness_pct" in self._settings:
            return round(self._settings["brightness_pct"])
        settings = self._sun_light_settings.get_settings(
            self.sleep_mode_switch.is_on,
            transition=0,
        )
        return round(settings["brightness_pct"])

    def preview_intelligence(self, lights: list[str] | None = None) -> dict[str, Any]:
        """Return read-only policy results for a service preview/explain request."""
        if not self._intelligence_enabled:
            return {}
        baseline = self._intelligence_baseline_brightness()
        selected_lights = lights or self.lights
        decisions = {
            light: self._intelligence_decision(light, baseline)
            for light in selected_lights
        }
        self._intelligence_decisions.update(decisions)
        return deepcopy(decisions)

    def _intelligence_target_brightness(
        self,
        light: str,
        baseline_brightness_pct: int,
    ) -> int | None:
        """Return an authorized active or opt-in shadow baseline target."""
        if not self._intelligence_enabled:
            return baseline_brightness_pct
        shadow_baseline = self._intelligence_shadow_baseline_brightness_active
        if self._intelligence_shadow_actuation_blocked and not shadow_baseline:
            return None
        external_turn_on = shadow_baseline and not is_on(self.hass, light)
        decision = self._intelligence_decision(
            light,
            baseline_brightness_pct,
            external_turn_on=external_turn_on,
        )
        self._intelligence_decisions[light] = decision
        target = int(decision["target_brightness_pct"])
        if shadow_baseline:
            # The room estimate refines the tanh curve; it does not escape the
            # configured Adaptive Lighting envelope. This is especially
            # important on an external turn-on, where an estimate near zero
            # could otherwise power a dimmable fixture at an unusable level.
            target = max(
                round(self._minimum_brightness),
                min(baseline_brightness_pct, target),
            )
        if external_turn_on:
            # The caller owns the turn-on. We add only a bounded brightness
            # target to the same request, even though intelligence itself has
            # no power-state authority in shadow mode.
            return target if decision["can_adjust"] else None
        permitted = (
            decision["can_adjust"]
            if is_on(self.hass, light)
            else decision["can_turn_on"]
        )
        if not permitted:
            return None
        return target

    @property
    def icon(self) -> str:
        """Icon to use in the frontend, if any."""
        return self._icon

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return the attributes of the switch."""
        extra_state_attributes: dict[str, Any] = {"configuration": self._config}
        if self._behavior_runtime is not None:
            behavior = self._behavior_runtime.diagnostics
            sample_counts = behavior.get("sample_counts")
            if isinstance(sample_counts, Mapping):
                behavior["sample_counts"] = dict(
                    sorted(sample_counts.items())[:MAX_STATE_SAMPLE_COUNTS],
                )
                behavior["sample_counts_truncated"] = (
                    len(sample_counts) > MAX_STATE_SAMPLE_COUNTS
                )
            decisions = behavior.get("last_decisions", [])
            behavior["shadow_ready"] = sum(
                bool(item.get("ready"))
                for item in decisions
                if isinstance(item, Mapping)
            )
            behavior["shadow_executed"] = sum(
                bool(item.get("executed"))
                for item in decisions
                if isinstance(item, Mapping)
            )
            behavior["actuation_enabled"] = self._behavior_actuation_enabled()
            extra_state_attributes["intelligence_behavior"] = behavior
        if not self.is_on:
            for key in self._settings:
                extra_state_attributes[key] = None
            return _fit_state_attribute_budget(extra_state_attributes)
        extra_state_attributes["manual_control"] = [
            light for light in self.lights if self.manager.manual_control.get(light)
        ]
        extra_state_attributes.update(self._settings)
        if self._intelligence_enabled:
            decisions = self._intelligence_decisions
            exposed_decisions = dict(
                sorted(decisions.items())[:MAX_STATE_INTELLIGENCE_DECISIONS],
            )
            extra_state_attributes["intelligence"] = {
                "enabled": True,
                "configured_shadow_mode": self._intelligence_shadow_mode,
                "shadow_mode": self._intelligence_shadow_actuation_blocked,
                "shadow_baseline_brightness_configured": (
                    self._intelligence_shadow_baseline_brightness
                ),
                "shadow_baseline_brightness_active": (
                    self._intelligence_shadow_baseline_brightness_active
                ),
            }
            extra_state_attributes["intelligence_decisions"] = deepcopy(
                exposed_decisions,
            )
            extra_state_attributes["intelligence_decisions_truncated"] = (
                len(decisions) > MAX_STATE_INTELLIGENCE_DECISIONS
            )
            if self._training is not None:
                extra_state_attributes["intelligence_training"] = (
                    self._training.summary()
                )
            if self._discovery_snapshot is not None:
                snapshot = self._discovery_snapshot
                extra_state_attributes["intelligence_discovery"] = {
                    "revision": snapshot.revision,
                    "reason": snapshot.reason,
                    "areas": list(snapshot.monitored_area_ids),
                    "entity_count": len(snapshot.entities),
                    "entities": [
                        {
                            "entity_id": item.entity_id,
                            "area": item.area_name or item.area_id,
                            "capabilities": list(item.capabilities),
                            "status": item.status,
                            "explicit_controlled": item.explicit_controlled,
                        }
                        for item in snapshot.entities[:MAX_STATE_DIAGNOSTIC_ITEMS]
                    ],
                    "added": [
                        item.entity_id
                        for item in snapshot.added[:MAX_STATE_DIAGNOSTIC_ITEMS]
                    ],
                    "removed": [
                        item.entity_id
                        for item in snapshot.removed[:MAX_STATE_DIAGNOSTIC_ITEMS]
                    ],
                    "moved": [
                        item.to_dict()
                        for item in snapshot.moved[:MAX_STATE_DIAGNOSTIC_ITEMS]
                    ],
                    "renamed": [
                        item.to_dict()
                        for item in snapshot.renamed[:MAX_STATE_DIAGNOSTIC_ITEMS]
                    ],
                    "added_truncated": (
                        len(snapshot.added) > MAX_STATE_DIAGNOSTIC_ITEMS
                    ),
                    "removed_truncated": (
                        len(snapshot.removed) > MAX_STATE_DIAGNOSTIC_ITEMS
                    ),
                    "moved_truncated": (
                        len(snapshot.moved) > MAX_STATE_DIAGNOSTIC_ITEMS
                    ),
                    "renamed_truncated": (
                        len(snapshot.renamed) > MAX_STATE_DIAGNOSTIC_ITEMS
                    ),
                    "truncated": (len(snapshot.entities) > MAX_STATE_DIAGNOSTIC_ITEMS),
                }
        timers = self.manager.auto_reset_manual_control_timers
        extra_state_attributes["autoreset_time_remaining"] = {
            light: time
            for light in self.lights
            if (timer := timers.get(light)) and (time := timer.remaining_time()) > 0
        }
        return _fit_state_attribute_budget(extra_state_attributes)

    def create_context(
        self,
        which: str = "default",
        parent: Context | None = None,
    ) -> Context:
        """Create a context that identifies this Adaptive Lighting instance."""
        context = create_context(self._name, which, self._context_cnt, parent=parent)
        self._context_cnt += 1
        return context

    async def async_turn_on(  # type: ignore[override]
        self,
        adapt_lights: bool = True,
    ) -> None:
        """Turn on adaptive lighting."""
        _LOGGER.debug(
            "%s: Called 'async_turn_on', current state is '%s'",
            self._name,
            self._state,
        )
        if self.is_on and (
            self._behavior_runtime is None or self._behavior_runtime_started
        ):
            return
        self._state = True
        await self._async_start_behavior_runtime()
        self._schedule_behavior_evaluation()
        self.manager.reset(*self.lights)
        await self._setup_listeners()
        if adapt_lights:
            await self._update_attrs_and_maybe_adapt_lights(
                context=self.create_context("turn_on"),
                transition=self.initial_transition,
                force=True,
            )

    async def async_turn_off(self, **kwargs: Any) -> None:  # noqa: ARG002
        """Turn off adaptive lighting."""
        if not self.is_on and not self._behavior_runtime_started:
            return
        self._state = False
        self._behavior_runtime_started = False
        behavior_task = self._cancel_behavior_evaluation()
        self._remove_listeners()
        await self._async_stop_behavior_runtime(behavior_task)
        self.manager.reset(*self.lights)

    async def _async_update_at_interval_action(
        self,
        now: Any = None,  # noqa: ARG002
    ) -> None:
        """Update the attributes and maybe adapt the lights."""
        await self._update_attrs_and_maybe_adapt_lights(
            context=self.create_context("interval"),
            transition=self._transition,
            force=False,
        )

    async def prepare_adaptation_data(
        self,
        light: str,
        transition: int | None = None,
        adapt_brightness: bool | None = None,
        adapt_color: bool | None = None,
        prefer_rgb_color: bool | None = None,
        force: bool = False,
        context: Context | None = None,
    ) -> AdaptationData | None:
        """Prepare `AdaptationData` for adapting a light."""
        adaptation_attributes = self.manager.get_adaption_control_attributes(
            self,
            light,
        )

        if transition is None:
            transition = self._transition
        if adapt_brightness is None:
            adapt_brightness = (
                LightControlAttributes.BRIGHTNESS in adaptation_attributes
            )
        if adapt_color is None:
            adapt_color = LightControlAttributes.COLOR in adaptation_attributes
        if self._intelligence_shadow_baseline_brightness_active:
            # Shadow baseline is brightness-only by contract, even if a stale
            # restored color switch or an explicit service parameter requests
            # color adaptation.
            adapt_color = False
        if prefer_rgb_color is None:
            prefer_rgb_color = self._prefer_rgb_color

        if not adapt_color and not adapt_brightness:
            _LOGGER.debug(
                "%s: Skipping adaptation of %s because both adapt_brightness and"
                " adapt_color are False",
                self._name,
                light,
            )
            return None

        # The switch might be off and not have _settings set.
        self._settings = self._sun_light_settings.get_settings(
            self.sleep_mode_switch.is_on,
            transition,
        )

        # Build service data.
        service_data: dict[str, Any] = {ATTR_ENTITY_ID: light}
        features = _supported_features(self.hass, light)

        # Check transition == 0 to fix #378
        use_transition = "transition" in features and transition > 0
        if use_transition:
            service_data[ATTR_TRANSITION] = transition

        if "brightness" in features and adapt_brightness:
            brightness_pct = self._intelligence_target_brightness(
                light,
                round(self._settings["brightness_pct"]),
            )
            if brightness_pct is not None:
                brightness = round(255 * brightness_pct / 100)
                service_data[ATTR_BRIGHTNESS] = brightness

        sleep_rgb = (
            self.sleep_mode_switch.is_on
            and self._sun_light_settings.sleep_rgb_or_color_temp == "rgb_color"
        )
        if (
            "color_temp" in features
            and adapt_color
            and not (prefer_rgb_color and "color" in features)
            and not (sleep_rgb and "color" in features)
            and not (self._settings["force_rgb_color"] and "color" in features)
        ):
            _LOGGER.debug("%s: Setting color_temp of light %s", self._name, light)
            state = self.hass.states.get(light)
            assert isinstance(state, State)
            attributes = state.attributes
            min_kelvin = attributes["min_color_temp_kelvin"]
            max_kelvin = attributes["max_color_temp_kelvin"]
            color_temp_kelvin = self._settings["color_temp_kelvin"]
            color_temp_kelvin = clamp(color_temp_kelvin, min_kelvin, max_kelvin)
            service_data[ATTR_COLOR_TEMP_KELVIN] = color_temp_kelvin
        elif "color" in features and adapt_color:
            _LOGGER.debug("%s: Setting rgb_color of light %s", self._name, light)
            service_data[ATTR_RGB_COLOR] = self._settings["rgb_color"]

        required_attrs = [ATTR_RGB_COLOR, ATTR_COLOR_TEMP_KELVIN, ATTR_BRIGHTNESS]
        if not any(attr in service_data for attr in required_attrs):
            _LOGGER.debug(
                "%s: Skipping adaptation of %s because no relevant attributes"
                " are set in service_data: %s",
                self._name,
                light,
                service_data,
            )
            return None

        context = context or self.create_context("adapt_lights")

        return prepare_adaptation_data(
            self.hass,
            light,
            context,
            transition if use_transition else 0,
            self._send_split_delay / 1000.0,
            service_data,
            split=self._separate_turn_on_commands,
            filter_by_state=self._skip_redundant_commands,
            force=force,
        )

    async def _adapt_light(  # noqa: PLR0911
        self,
        light: str,
        context: Context,
        transition: int | None = None,
        adapt_brightness: bool | None = None,
        adapt_color: bool | None = None,
        prefer_rgb_color: bool | None = None,
        force: bool = False,
    ) -> None:
        shadow_baseline = self._intelligence_shadow_baseline_brightness_active
        if self._intelligence_shadow_actuation_blocked and not shadow_baseline:
            # A commissioned shadow instance may calculate and expose policy
            # decisions, but it must never issue an Adaptive Lighting light call.
            return
        if shadow_baseline and not is_on(self.hass, light):
            # Autonomous interval/reactive paths must never use a brightness
            # service call to power on an off light. External turn-on requests
            # use the interceptor path and never reach this branch.
            return
        if shadow_baseline and self._intelligence_snapshot()["manual_hold"]:
            return
        if (
            self._intelligence_enabled
            and not self._intelligence_shadow_actuation_blocked
        ):
            snapshot = self._intelligence_snapshot()
            if snapshot["manual_hold"]:
                # An external/manual helper is a whole-light hold. Attribute-
                # level Adaptive Lighting takeover remains enforced below by
                # the existing manager machinery.
                return
            if not is_on(self.hass, light):
                baseline = self._intelligence_baseline_brightness()
                decision = self._intelligence_decision(light, baseline)
                self._intelligence_decisions[light] = decision
                if not decision["can_turn_on"]:
                    return
        if (lock := self.manager.turn_off_locks.get(light)) and lock.locked():
            _LOGGER.debug("%s: '%s' is locked", self._name, light)
            return

        data = await self.prepare_adaptation_data(
            light,
            transition,
            adapt_brightness,
            False if shadow_baseline else adapt_color,
            prefer_rgb_color,
            force,
            context,
        )
        if data is None:
            return  # nothing to adapt

        await self.execute_cancellable_adaptation_calls(data)

    async def _execute_adaptation_calls(self, data: AdaptationData) -> None:
        """Executes a sequence of adaptation service calls for the given service datas."""
        if (
            self._intelligence_shadow_actuation_blocked
            and not self._intelligence_shadow_baseline_brightness_active
        ):
            return
        for index in range(data.max_length):
            is_first_call = index == 0

            # Sleep between multiple service calls.
            if not is_first_call or data.initial_sleep:
                await asyncio.sleep(data.sleep_time)

            # Instead of directly iterating the generator in the while-loop, we get
            # the next item here after the sleep to make sure it incorporates state
            # changes which happened during the sleep.
            service_data = await data.next_service_call_data()

            if not service_data:
                # All service datas processed
                break

            if (
                not data.force
                and not is_on(self.hass, data.entity_id)
                # if proactively adapting, we are sure that it came from a `light.turn_on`
                and not self.manager.is_proactively_adapting(data.context.id)
            ):
                # Do a last-minute check if the entity is still on.
                _LOGGER.debug(
                    "%s: Skipping adaptation of %s because it is now off",
                    self._name,
                    data.entity_id,
                )
                return

            _LOGGER.debug(
                "%s: Scheduling 'light.turn_on' with the following 'service_data': %s"
                " with context.id='%s'",
                self._name,
                service_data,
                data.context.id,
            )
            light = service_data[ATTR_ENTITY_ID]
            self.manager.last_service_data[light] = {
                **self.manager.last_service_data.get(light, {}),
                **service_data,
            }
            await self.hass.services.async_call(
                LIGHT_DOMAIN,
                SERVICE_TURN_ON,
                service_data,
                context=data.context,
            )

    async def execute_cancellable_adaptation_calls(
        self,
        data: AdaptationData,
    ) -> None:
        """Executes a cancellable sequence of adaptation service calls for the given service datas.

        Wraps the sequence of service calls in a task that can be cancelled from elsewhere, e.g.,
        to cancel an ongoing adaptation when a light is turned off.
        """
        # Prevent overlap of multiple adaptation sequences
        self.manager.cancel_ongoing_adaptation_calls(data.entity_id)
        _LOGGER.debug(
            "%s: execute_cancellable_adaptation_calls with data: %s",
            self._name,
            data,
        )
        # Execute adaptation calls within a task
        try:
            task = asyncio.ensure_future(self._execute_adaptation_calls(data))
            if LightControlAttributes.BRIGHTNESS in data.attributes:
                self.manager.adaptation_tasks_brightness[data.entity_id] = task
            if LightControlAttributes.COLOR in data.attributes:
                self.manager.adaptation_tasks_color[data.entity_id] = task
            await task
        except asyncio.CancelledError:
            _LOGGER.debug(
                "%s: Ongoing adaptation of %s cancelled, with AdaptationData: %s",
                self._name,
                data.entity_id,
                data,
            )

    async def _update_attrs_and_maybe_adapt_lights(
        self,
        *,
        context: Context,
        lights: list[str] | None = None,
        transition: int | None = None,
        force: bool = False,
    ) -> None:
        assert context is not None
        _LOGGER.debug(
            "%s: '_update_attrs_and_maybe_adapt_lights' called with context.id='%s'"
            " lights: '%s', transition: '%s', force: '%s'",
            self._name,
            context.id,
            lights,
            transition,
            force,
        )
        assert self.is_on
        self._settings.update(
            self._sun_light_settings.get_settings(
                self.sleep_mode_switch.is_on,
                transition,
            ),
        )
        # Shadow mode deliberately refreshes both the existing sun baseline and
        # the intelligence decision before writing state, but the actuation gate
        # below prevents every Adaptive Lighting-originated light service call.
        self._refresh_intelligence_decisions()
        self.async_write_ha_state()

        if not force and self._only_once:
            return

        if lights is None:
            lights = self.lights

        on_lights = [light for light in lights if is_on(self.hass, light)]

        if force:
            filtered_lights = on_lights
        else:
            filtered_lights: list[str] = []
            for light in on_lights:
                # Don't adapt lights that haven't finished prior transitions.
                timer = self.manager.transition_timers.get(light)
                if timer is not None and timer.is_running():
                    _LOGGER.debug(
                        "%s: Light '%s' is still transitioning, context.id='%s'",
                        self._name,
                        light,
                        context.id,
                    )
                elif (
                    # This is to prevent lights immediately turning on after
                    # being turned off in 'interval' update, see #726
                    not self._detect_non_ha_changes
                    and is_our_context(context, "interval")
                    and (turn_on := self.manager.turn_on_event.get(light))
                    and (turn_off := self.manager.turn_off_event.get(light))
                    and turn_off.time_fired > turn_on.time_fired
                ):
                    _LOGGER.debug(
                        "%s: Light '%s' was turned just turned off, context.id='%s'",
                        self._name,
                        light,
                        context.id,
                    )
                else:
                    filtered_lights.append(light)

        _LOGGER.debug("%s: filtered_lights: '%s'", self._name, filtered_lights)
        if not filtered_lights:
            return

        tasks: list[asyncio.Task[None]] = []
        for light in filtered_lights:
            await self.manager.update_manually_controlled_from_untracked_change(
                self,
                light,
                force,
                context,
            )

            # Performance optimization: Skip adaptation task if all attributes are
            # manually controlled and the task wouldn't actually do anything.
            if self.manager.get_adaption_control_attributes(self, light).has_none():
                _LOGGER.debug(
                    "%s: '%s' is being manually controlled, skip adaptation, context.id=%s.",
                    self._name,
                    light,
                    context.id,
                )
                continue

            _LOGGER.debug(
                "%s: Calling _adapt_light from _update_attrs_and_maybe_adapt_lights:"
                " '%s' with transition %s and context.id=%s",
                self._name,
                light,
                transition,
                context.id,
            )
            coro = self._adapt_light(light, context, transition, force=force)
            task = self.hass.async_create_task(
                coro,
            )
            tasks.append(task)
        if tasks:
            await asyncio.gather(*tasks)

    async def _respond_to_off_to_on_event(
        self,
        entity_id: str,
        event: Event[EventStateChangedData],
    ) -> None:
        assert not self.manager.is_proactively_adapting(event.context.id)
        from_turn_on = self.manager._off_to_on_state_event_is_from_turn_on(
            entity_id,
            event,
        )
        if (
            self._take_over_control
            and not self._detect_non_ha_changes
            and not from_turn_on
        ):
            # There is an edge case where 2 switches control the same light, e.g.,
            # one for brightness and one for color. Now we will mark both switches
            # as manually controlled, which is not 100% correct.
            _LOGGER.debug(
                "%s: Ignoring 'off' → 'on' event for '%s' with context.id='%s'"
                " because 'light.turn_on' was not called by HA and"
                " 'detect_non_ha_changes' is False",
                self._name,
                entity_id,
                event.context.id,
            )
            self.manager.set_manual_control_attributes(entity_id)
            return

        if (
            self._take_over_control
            and self._adapt_only_on_bare_turn_on
            and from_turn_on
            # adaptive_lighting.apply can turn on light, so check this is not our context
            and not is_our_context(event.context)
        ):
            service_data = self.manager.turn_on_event[entity_id].data[ATTR_SERVICE_DATA]
            if self.manager._mark_manual_control_if_non_bare_turn_on(
                entity_id,
                service_data,
            ):
                _LOGGER.debug(
                    "Marked attributes from service_data as manually controlled for '%s' "
                    "with context.id='%s'. Continuing to adapt remaining attributes. "
                    "service_data: '%s'",
                    entity_id,
                    event.context.id,
                    service_data,
                )

        if self._adapt_delay > 0:
            await asyncio.sleep(self._adapt_delay)

        await self._update_attrs_and_maybe_adapt_lights(
            context=self.create_context("light_event", parent=event.context),
            lights=[entity_id],
            transition=self.initial_transition,
            force=True,
        )

    async def _sleep_mode_switch_state_event_action(
        self,
        event: Event[EventStateChangedData],
    ) -> None:
        if not _is_state_event(event, (STATE_ON, STATE_OFF)):
            _LOGGER.debug("%s: Ignoring sleep event %s", self._name, event)
            return
        _LOGGER.debug(
            "%s: _sleep_mode_switch_state_event_action, event: '%s'",
            self._name,
            event,
        )
        # Reset the manually controlled status when the "sleep mode" changes
        self.manager.reset(*self.lights)
        await self._update_attrs_and_maybe_adapt_lights(
            context=self.create_context("sleep", parent=event.context),
            transition=self._sleep_transition,
            force=True,
        )

    def fire_manual_control_event(
        self,
        light: str,
        context: Context,
    ) -> None:
        """Fire an event that 'light' is marked as manual_control."""
        _LOGGER.debug(
            "'adaptive_lighting.manual_control' event fired for %s for light %s",
            self.entity_id,
            light,
        )
        manual_attributes = self.manager.get_manual_control_attributes(light)
        self.hass.bus.async_fire(
            f"{DOMAIN}.manual_control",
            {
                ATTR_ENTITY_ID: light,
                SWITCH_DOMAIN: self.entity_id,
                CONF_MANUAL_CONTROL: manual_attributes,
            },
            context=context,
        )
        self._schedule_training_sample(light, context, manual_attributes)

    def _schedule_training_sample(
        self,
        light: str,
        context: Context,
        manual_attributes: LightControlAttributes,
    ) -> None:
        """Offer a durable, attributable brightness correction to local training."""
        if (
            self._training is None
            or not manual_attributes & LightControlAttributes.BRIGHTNESS
        ):
            return
        state = self.hass.states.get(light)
        brightness = (
            state.attributes.get(ATTR_BRIGHTNESS) if state is not None else None
        )
        if not isinstance(brightness, (int, float)):
            return
        selected = round(100 * brightness / 255)
        snapshot = self._intelligence_snapshot()
        current_decision = self._intelligence_decisions.get(light, {})
        baseline = current_decision.get(
            "target_brightness_pct",
            self._intelligence_baseline_brightness(),
        )
        intent = current_decision.get(
            "intent",
            snapshot["intent_hint"],
        )
        time_bucket, daylight_band = self._intelligence_learning_dimensions(snapshot)
        source = (
            "user"
            if context.user_id is not None
            else "automation"
            if context.parent_id is not None
            else "physical"
        )
        self.hass.async_create_task(
            self._training.async_ingest_sample(
                {
                    "zone": self._name,
                    "baseline": baseline,
                    "selected": selected,
                    "source": source,
                    "intent": intent,
                    "time_bucket": time_bucket,
                    "daylight_band": daylight_band,
                    "safety_context": bool(
                        snapshot.get("emergency") or snapshot.get("sleep"),
                    ),
                    "hard_cap": bool(
                        snapshot.get("emergency") or snapshot.get("sleep"),
                    ),
                    "observed_at": dt_util.utcnow().isoformat(),
                    "metadata": {
                        "context_has_parent": context.parent_id is not None,
                        "context_has_user": context.user_id is not None,
                    },
                },
            ),
            name=f"adaptive_lighting_training_{slugify(self._name)}",
        )

    @staticmethod
    def _intelligence_learning_dimensions(snapshot: dict[str, Any]) -> tuple[str, str]:
        """Return coarse, privacy-preserving temporal and daylight buckets."""
        hour = dt_util.as_local(dt_util.utcnow()).hour
        time_bucket = (
            "morning"
            if 5 <= hour < 12
            else "day"
            if 12 <= hour < 17
            else "evening"
            if 17 <= hour < 23
            else "night"
        )
        illuminance = snapshot.get("illuminance_value")
        daylight_band = (
            "unknown"
            if not isinstance(illuminance, (int, float))
            else "dark"
            if illuminance < 30
            else "dim"
            if illuminance < 150
            else "bright"
        )
        return time_bucket, daylight_band


class SimpleSwitch(SwitchEntity, RestoreEntity):
    """Representation of a Adaptive Lighting switch."""

    def __init__(
        self,
        which: str,
        initial_state: bool,
        hass: HomeAssistant,
        config_entry: ConfigEntry,
        icon: str,
        restore_state: bool = True,
    ) -> None:
        """Initialize the Adaptive Lighting switch."""
        self.hass = hass
        data = validate(config_entry)
        self._icon = icon
        self._state: bool = initial_state
        self._which = which
        self._config_name = data[CONF_NAME]
        self._unique_id = f"{self._config_name}_{slugify(self._which)}"
        self._name = f"Adaptive Lighting {which}: {self._config_name}"
        self._initial_state = initial_state
        self._restore_state = restore_state

    @property
    def name(self) -> str:
        """Return the name of the device if any."""
        return self._name

    @property
    def unique_id(self) -> str:
        """Return the unique ID of entity."""
        return self._unique_id

    @property
    def icon(self) -> str:
        """Icon to use in the frontend, if any."""
        return self._icon

    @property
    def is_on(self) -> bool | None:
        """Return true if adaptive lighting is on."""
        return self._state

    @property
    def device_info(self) -> DeviceInfo:
        """Return the device info, used to group this and adjacent entities in the UI."""
        return DeviceInfo(
            identifiers={
                (DOMAIN, self._config_name),
            },
            name=f"Adaptive Lighting: {self._config_name}",
            entry_type=DeviceEntryType.SERVICE,
        )

    async def async_added_to_hass(self) -> None:
        """Call when entity about to be added to hass."""
        if not self._restore_state:
            if self._initial_state:
                await self.async_turn_on()
            else:
                await self.async_turn_off()
            return
        last_state = await self.async_get_last_state()
        _LOGGER.debug("%s: last state is %s", self._name, last_state)
        if (last_state is None and self._initial_state) or (
            last_state is not None and last_state.state == STATE_ON
        ):
            await self.async_turn_on()
        else:
            await self.async_turn_off()

    async def async_turn_on(self, **kwargs: Any) -> None:  # noqa: ARG002
        """Turn on adaptive lighting sleep mode."""
        _LOGGER.debug("%s: Turning on", self._name)
        self._state = True

    async def async_turn_off(self, **kwargs: Any) -> None:  # noqa: ARG002
        """Turn off adaptive lighting sleep mode."""
        _LOGGER.debug("%s: Turning off", self._name)
        self._state = False


AdaptiveSwitches = list[AdaptiveSwitch]
AdaptiveSwitchMap = dict[AdaptiveSwitch, list[str]]


class AdaptiveLightingManager:
    """Track 'light.turn_off' and 'light.turn_on' service calls."""

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize the AdaptiveLightingManager that is shared among all switches."""
        assert hass is not None
        self.hass = hass
        self.lights: set[str] = set()

        # Tracks 'light.turn_off' service calls
        self.turn_off_event: dict[str, Event] = {}
        # Tracks 'light.turn_on' service calls
        self.turn_on_event: dict[str, Event] = {}
        # Tracks 'light.toggle' service calls
        self.toggle_event: dict[str, Event] = {}
        # Tracks 'on' → 'off' state changes
        self.on_to_off_event: dict[str, Event[EventStateChangedData]] = {}
        # Tracks 'off' → 'on' state changes
        self.off_to_on_event: dict[str, Event[EventStateChangedData]] = {}
        # Keep 'asyncio.sleep' tasks that can be cancelled by 'light.turn_on' events
        self.sleep_tasks: dict[str, asyncio.Task[None]] = {}
        # Locks that prevent light adjusting when waiting for a light to 'turn_off'
        self.turn_off_locks: dict[str, asyncio.Lock] = {}
        # Tracks which lights are manually controlled
        self.manual_control: dict[str, LightControlAttributes] = {}
        # Track 'state_changed' events of self.lights resulting from this integration
        self.our_last_state_on_change: dict[str, list[State]] = {}
        # Track last 'service_data' to 'light.turn_on' resulting from this integration
        self.last_service_data: dict[str, dict[str, Any]] = {}
        # Track ongoing split adaptations to be able to cancel them
        self.adaptation_tasks_brightness: dict[str, asyncio.Task[None]] = {}
        self.adaptation_tasks_color: dict[str, asyncio.Task[None]] = {}

        # Track auto reset of manual_control
        self.auto_reset_manual_control_timers: dict[str, _AsyncSingleShotTimer] = {}
        self.auto_reset_manual_control_times: dict[str, float] = {}

        # Track light transitions
        self.transition_timers: dict[str, _AsyncSingleShotTimer] = {}

        # Track _execute_cancellable_adaptation_calls tasks
        self.adaptation_tasks: set[asyncio.Task[None]] = set()

        # Setup listeners and its callbacks to remove them later
        self.listener_removers = [
            self.hass.bus.async_listen(
                EVENT_CALL_SERVICE,
                self.turn_on_off_event_listener,
            ),
            self.hass.bus.async_listen(
                EVENT_STATE_CHANGED,
                self.state_changed_event_listener,
            ),
        ]

        self._proactively_adapting_contexts: dict[str, str] = {}
        self._context_cnt: int = 0

        try:
            self.listener_removers.append(
                setup_service_call_interceptor(
                    hass,
                    LIGHT_DOMAIN,
                    SERVICE_TURN_ON,
                    self._service_interceptor_turn_on_handler,
                ),
            )

            self.listener_removers.append(
                setup_service_call_interceptor(
                    hass,
                    LIGHT_DOMAIN,
                    SERVICE_TOGGLE,
                    self._service_interceptor_turn_on_handler,
                ),
            )
        except RuntimeError:
            _LOGGER.warning(
                "Failed to set up service call interceptors, "
                "falling back to event-reactive mode",
                exc_info=True,
            )

    def disable(self) -> None:
        """Disable the listener by removing all subscribed handlers."""
        for remove in self.listener_removers:
            remove()

    def set_proactively_adapting(self, context_id: str, entity_id: str) -> None:
        """Declare the adaptation with context_id as proactively adapting,
        and associate it to an entity_id.
        """  # noqa: D205
        self._proactively_adapting_contexts[context_id] = entity_id

    def is_proactively_adapting(self, context_id: str) -> bool:
        """Determine whether an adaptation with the given context_id is proactive."""
        is_proactively_adapting_context = (
            context_id in self._proactively_adapting_contexts
        )

        _LOGGER.debug(
            "is_proactively_adapting_context='%s', context_id='%s'",
            is_proactively_adapting_context,
            context_id,
        )

        return is_proactively_adapting_context

    def clear_proactively_adapting(self, entity_id: str) -> None:
        """Clear all context IDs associated with the given entity ID.

        Call this method to clear past context IDs and avoid a memory leak.
        """
        # First get the keys to avoid modifying the dict while iterating it
        keys = [
            k for k, v in self._proactively_adapting_contexts.items() if v == entity_id
        ]
        for key in keys:
            self._proactively_adapting_contexts.pop(key)

    def create_context(
        self,
        which: str = "default",
        parent: Context | None = None,
    ) -> Context:
        """Create a context that identifies this integration."""
        context = create_context("manager", which, self._context_cnt, parent=parent)
        self._context_cnt += 1
        return context

    def _separate_entity_ids(
        self,
        entity_ids: list[str],
        data: ServiceData,
    ) -> tuple[AdaptiveSwitchMap, list[str]]:
        # Create a mapping from switch to entity IDs
        # AdaptiveSwitch → entity_ids mapping
        switch_to_eids: AdaptiveSwitchMap = {}
        skipped: list[str] = []
        for entity_id in entity_ids:
            try:
                switch = _switch_with_lights(
                    self.hass,
                    [entity_id],
                    # Do not expand light groups, because HA will make a separate light.turn_on
                    # call where the lights are expanded, and that call will be intercepted.
                    expand_light_groups=False,
                )
            except NoSwitchFoundError:
                # Needs to make the original call but without adaptation
                skipped.append(entity_id)
                _LOGGER.debug(
                    "No switch found for entity_id='%s', skipped='%s'",
                    entity_id,
                    skipped,
                )
            else:
                if (
                    not switch.is_on
                    or not switch._intercept
                    or (
                        switch._intelligence_shadow_actuation_blocked
                        and not switch._intelligence_shadow_baseline_brightness_active
                    )
                    # Never adapt on light groups, because HA will make a separate light.turn_on
                    or ((e := self.hass.states.get(entity_id)) and _is_light_group(e))
                    # Prevent adaptation of TURN_ON calls when light is already on,
                    # and of TOGGLE calls when toggling off.
                    or self.hass.states.is_state(entity_id, STATE_ON)
                    or self.manual_control.get(entity_id, False)
                    or (
                        switch._take_over_control
                        and switch._adapt_only_on_bare_turn_on
                        and self._mark_manual_control_if_non_bare_turn_on(
                            entity_id,
                            data[CONF_PARAMS],
                        )
                    )
                ):
                    _LOGGER.debug(
                        "Switch is off or light is already on for entity_id='%s', skipped='%s'"
                        " (is_on='%s', is_state='%s', manual_control='%s', switch._intercept='%s')",
                        entity_id,
                        skipped,
                        switch.is_on,
                        self.hass.states.is_state(entity_id, STATE_ON),
                        self.manual_control.get(entity_id, False),
                        switch._intercept,
                    )
                    skipped.append(entity_id)
                else:
                    switch_to_eids.setdefault(switch, []).append(entity_id)
        return switch_to_eids, skipped

    def _correct_for_multi_light_intercept(
        self,
        entity_ids: list[str],
        switch_to_eids: AdaptiveSwitchMap,
        skipped: list[str],
    ):
        # Check for `multi_light_intercept: true/false`
        mli = [sw._multi_light_intercept for sw in switch_to_eids]
        more_than_one_switch = len(switch_to_eids) > 1
        single_switch_with_multiple_lights = (
            len(switch_to_eids) == 1 and len(next(iter(switch_to_eids.values()))) > 1
        )
        switch_without_multi_light_intercept = not all(mli)
        if more_than_one_switch and switch_without_multi_light_intercept:
            _LOGGER.warning(
                "Multiple switches (%s) targeted, but not all have"
                " `multi_light_intercept: true`, so skipping intercept"
                " for all lights.",
                switch_to_eids,
            )
            skipped = entity_ids
            switch_to_eids = {}
        elif (
            single_switch_with_multiple_lights and switch_without_multi_light_intercept
        ):
            _LOGGER.warning(
                "Single switch with multiple lights targeted (%s), but"
                " `multi_light_intercept: true` is not set, so skipping intercept"
                " for all lights.",
                switch_to_eids,
            )
            skipped = entity_ids
            switch_to_eids = {}
        return switch_to_eids, skipped

    async def _service_interceptor_turn_on_handler(
        self,
        call: ServiceCall,
        service_data: ServiceData,
    ) -> None:
        """Intercept `light.turn_on` and `light.toggle` service calls and adapt them.

        It is possible that the calls are made for multiple lights at once,
        which in turn might be in different switches or no switches at all.
        If there are lights that are not all in a single switch, we need to
        make multiple calls to `light.turn_on` with the correct entity IDs.
        One of these calls can be intercepted and adapted, the others need to
        be adapted by calling `_adapt_light` with the correct entity IDs or
        by calling `light.turn_on` directly.

        We create a mapping from switch to entity IDs and keep a list
        of skipped lights which are lights in no switches or in switches that
        are off or lights that are already on.

        If there is only one switch and 0 skipped lights, we just intercept the
        call directly.

        If there are multiple switches and skipped lights, we can adapt the call
        for one of the switches to include only the lights in that switch and
        need to call `_adapt_light` for the other switches with their
        entity_ids. For skipped lights, we call light.turn_on directly with the
        entity_ids and original service data.

        If there are only skipped lights, we can use the intercepted call
        directly.
        """
        is_skipped_hash = is_our_context(call.context, "skipped")
        _LOGGER.debug(
            "(0) _service_interceptor_turn_on_handler: call.context.id='%s', is_skipped_hash='%s'",
            call.context.id,
            is_skipped_hash,
        )
        if is_our_context(call.context) and not is_skipped_hash:
            # Don't adapt our own service calls, but do re-adapt calls that
            # were skipped by us
            return

        if has_effect_attribute(service_data[CONF_PARAMS]):
            return

        _LOGGER.debug(
            "(1) _service_interceptor_turn_on_handler: call='%s', service_data='%s'",
            call,
            service_data,
        )

        # Because `_service_interceptor_turn_on_single_light_handler` modifies the
        # original service data, we need to make a copy of it to use in the `skipped` call
        service_data_copy = deepcopy(service_data)

        entity_ids = self._get_entity_list(service_data)
        # Note: we do not expand light groups anywhere in this method, instead
        # we skip them and rely on the followup call that HA will make
        # with the expanded entity IDs.

        switch_to_eids, skipped = self._separate_entity_ids(
            entity_ids,
            service_data,
        )

        (
            switch_to_eids,
            skipped,
        ) = self._correct_for_multi_light_intercept(
            entity_ids,
            switch_to_eids,
            skipped,
        )
        _LOGGER.debug(
            "(2) _service_interceptor_turn_on_handler: switch_to_eids='%s', skipped='%s'",
            switch_to_eids,
            skipped,
        )

        def modify_service_data(
            service_data: ServiceData,
            entity_ids: list[str],
        ) -> dict[str, Any]:
            """Modify the service data to contain the entity IDs."""
            service_data.pop(ATTR_ENTITY_ID, None)
            service_data.pop(ATTR_AREA_ID, None)
            service_data[ATTR_ENTITY_ID] = entity_ids
            return service_data

        # Intercept the call for first switch and call _adapt_light for the rest
        has_intercepted = False  # Can only intercept a turn_on call once
        for switch, _entity_ids in switch_to_eids.items():
            transition = service_data[CONF_PARAMS].get(
                ATTR_TRANSITION,
                switch.initial_transition,
            )
            if not has_intercepted:
                _LOGGER.debug(
                    "(3) _service_interceptor_turn_on_handler: intercepting entity_ids='%s'",
                    _entity_ids,
                )
                await self._service_interceptor_turn_on_single_light_handler(
                    entity_ids=_entity_ids,
                    switch=switch,
                    transition=transition,
                    call=call,
                    data=modify_service_data(service_data, _entity_ids),
                )
                has_intercepted = True
                continue

            for eid in _entity_ids:
                # Must add a new context otherwise _adapt_light will bail out
                context = switch.create_context("intercept")
                self.clear_proactively_adapting(eid)
                self.set_proactively_adapting(context.id, eid)
                _LOGGER.debug(
                    "(4) _service_interceptor_turn_on_handler: calling `_adapt_light` with eid='%s', context='%s', transition='%s'",
                    eid,
                    context,
                    transition,
                )
                await switch._adapt_light(
                    light=eid,
                    context=context,
                    transition=transition,
                )

        # Call light.turn_on service for skipped entities
        if skipped:
            if not has_intercepted:
                assert set(skipped) == set(entity_ids)
                return  # The call will be intercepted with the original data
            # Call light turn_on service for skipped entities
            context = self.create_context("skipped")
            _LOGGER.debug(
                "(5) _service_interceptor_turn_on_handler: calling `light.turn_on` with skipped='%s', service_data: '%s', context='%s'",
                skipped,
                service_data_copy,  # This is the original service data
                context.id,
            )
            service_data = {ATTR_ENTITY_ID: skipped, **service_data_copy[CONF_PARAMS]}
            await self.hass.services.async_call(
                LIGHT_DOMAIN,
                SERVICE_TURN_ON,
                service_data,
                blocking=True,
                context=context,
            )

    async def _service_interceptor_turn_on_single_light_handler(
        self,
        entity_ids: list[str],
        switch: AdaptiveSwitch,
        transition: int,
        call: ServiceCall,
        data: ServiceData,
    ):
        _LOGGER.debug(
            "Intercepted TURN_ON call with data %s (%s)",
            data,
            call.context.id,
        )
        if (
            switch._intelligence_shadow_actuation_blocked
            and not switch._intelligence_shadow_baseline_brightness_active
        ):
            return

        # Reset because turning on the light, this also happens in
        # `state_changed_event_listener`, however, this function is called
        # before that one.
        self.reset(*entity_ids, reset_manual_control=False)
        for eid in entity_ids:
            self.clear_proactively_adapting(eid)

        adaptation_data = await switch.prepare_adaptation_data(
            entity_ids[0],
            transition,
        )
        if adaptation_data is None:
            return

        # Take first adaptation item to apply it to this service call
        first_service_data = await adaptation_data.next_service_call_data()

        if not first_service_data:
            return

        # Update/adapt service call data
        first_service_data.pop(ATTR_ENTITY_ID, None)
        # This is called as a preprocessing step by the schema validation of the original
        # service call and needs to be repeated here to also process the added adaptation data.
        # (A more generic alternative would be re-executing the validation, but that is more
        # complicated and unstable because it requires transformation of the data object back
        # into its original service call structure which cannot be reliably done due to the
        # lack of a bijective mapping.)
        preprocess_turn_on_alternatives(self.hass, first_service_data)
        data[CONF_PARAMS].update(first_service_data)

        # Schedule additional service calls for the remaining adaptation data.
        # We cannot know here whether there is another call to follow (since the
        # state can change until the next call), so we just schedule it and let
        # it sort out by itself.
        for entity_id in entity_ids:
            self.set_proactively_adapting(call.context.id, entity_id)
            self.set_proactively_adapting(adaptation_data.context.id, entity_id)
        adaptation_data.initial_sleep = True

        # Don't await to avoid blocking the service call.
        # Assign to a variable only to await in tests.
        self.adaptation_tasks.add(
            asyncio.create_task(
                switch.execute_cancellable_adaptation_calls(adaptation_data),
            ),
        )
        # Remove tasks that are done
        if done_tasks := [t for t in self.adaptation_tasks if t.done()]:
            self.adaptation_tasks.difference_update(done_tasks)

    def _handle_timer(
        self,
        light: str,
        timers_dict: dict[str, _AsyncSingleShotTimer],
        delay: float | None,
        reset_coroutine: Callable[[], Coroutine[Any, Any, None]],
    ) -> None:
        timer = timers_dict.get(light)
        if timer is not None:
            if delay is None:  # Timer object exists, but should not anymore
                timer.cancel()
                timers_dict.pop(light)
            else:  # Timer object already exists, just update the delay and restart it
                timer.delay = delay
                timer.start()
        elif delay is not None:  # Timer object does not exist, create it
            timer = _AsyncSingleShotTimer(delay, reset_coroutine)
            timers_dict[light] = timer
            timer.start()

    def start_transition_timer(self, light: str) -> None:
        """Mark a light as manually controlled."""
        last_service_data = self.last_service_data.get(light)
        if last_service_data is None:
            _LOGGER.debug(
                "No last service data for light %s, not starting timer.",
                light,
            )
            return

        last_transition = last_service_data.get(ATTR_TRANSITION)
        if not last_transition:
            _LOGGER.debug(
                "No transition in last adapt for light %s, not starting timer.",
                light,
            )
            return

        _LOGGER.debug(
            "Start transition timer of %s seconds for light %s",
            last_transition,
            light,
        )

        async def reset() -> None:
            # Called when the timer expires, doesn't need to do anything
            _LOGGER.debug(
                "Transition finished for light %s",
                light,
            )

        self._handle_timer(light, self.transition_timers, last_transition, reset)

    def set_auto_reset_manual_control_times(
        self,
        lights: list[str],
        time: float,
    ) -> None:
        """Set the time after which the lights are automatically reset."""
        if time == 0:
            return
        for light in lights:
            old_time = self.auto_reset_manual_control_times.get(light)
            if (old_time is not None) and (old_time != time):
                _LOGGER.info(
                    "Setting auto_reset_manual_control for '%s' from %s seconds to %s seconds."
                    " This might happen because the light is in multiple swiches"
                    " or because of a config change.",
                    light,
                    old_time,
                    time,
                )
            self.auto_reset_manual_control_times[light] = time

    def get_manual_control_attributes(
        self,
        light: str,
    ) -> LightControlAttributes:
        """Get the attributes for a light that are manually controlled."""
        return self.manual_control.get(light, LightControlAttributes.NONE)

    def set_manual_control_attributes(
        self,
        light: str,
        attributes: LightControlAttributes = LightControlAttributes.ALL,
    ) -> None:
        """Mark attributes of a light as manually controlled."""
        _LOGGER.debug(
            "Light %s: Setting manual control attributes to %s (from %s).",
            light,
            attributes,
            self.get_manual_control_attributes(light),
        )
        self.manual_control[light] = attributes
        delay = self.auto_reset_manual_control_times.get(light)

        async def reset() -> None:
            _LOGGER.debug(
                "Auto resetting 'manual_control' status of '%s' because"
                " it was not manually controlled for %s seconds.",
                light,
                delay,
            )
            self.reset(light)
            switches = _switches_with_lights(self.hass, [light])
            for switch in switches:
                if not switch.is_on:
                    continue
                await switch._update_attrs_and_maybe_adapt_lights(
                    context=switch.create_context("autoreset"),
                    lights=[light],
                    transition=switch.initial_transition,
                    force=True,
                )
            assert self.manual_control[light] == LightControlAttributes.NONE

        self._handle_timer(light, self.auto_reset_manual_control_timers, delay, reset)

    def add_manual_control_attributes(
        self,
        light: str,
        attributes: LightControlAttributes,
    ) -> None:
        """Add attributes to the manual control status of a light."""
        current = self.get_manual_control_attributes(light)
        _LOGGER.debug(
            "Light %s: Adding manual control attributes %s (current: %s).",
            light,
            attributes,
            current,
        )
        new = current | attributes
        self.set_manual_control_attributes(light, new)

    def get_adaption_control_attributes(
        self,
        switch: AdaptiveSwitch,
        light: str,
    ) -> LightControlAttributes:
        """Get the attributes that should be adapted for a light.

        Determines the attributes that should actually be adapted from the attributes
        marked as manually controlled, the state of adaptation switches, and the adaptation
        configuration.

        Example 1: When no attributes are marked as manually controlled and all adaptation
        switches are on, all attributes are returned.

        Example 2: When no attributes are marked as manually controlled and the brightness
        adaptation switch is off, only the color attribute is returned.

        Example 3: When only brightness is marked as manually controlled, but the configuration
        specifies to pause all adaptations on manual change, no attributes are returned so that
        color is also not adapted.
        """
        denied_adaptation_attributes = self.get_manual_control_attributes(light)

        if (
            denied_adaptation_attributes.has_any()
            and switch._take_over_control_mode == TakeOverControlMode.PAUSE_ALL
        ):
            # Extend to pausing all only if there is at least one manually controlled attribute
            denied_adaptation_attributes = LightControlAttributes.ALL

        enabled_adaptation_attributes = (
            LightControlAttributes.BRIGHTNESS
            if switch.adapt_brightness_switch.is_on
            else LightControlAttributes.NONE
        ) | (
            LightControlAttributes.COLOR
            if switch.adapt_color_switch.is_on
            else LightControlAttributes.NONE
        )

        return (
            LightControlAttributes.ALL
            & ~denied_adaptation_attributes
            & enabled_adaptation_attributes
        )

    def cancel_ongoing_adaptation_calls(
        self,
        light_id: str,
    ) -> None:
        """Cancel ongoing adaptation service calls for a specific light entity."""
        brightness_task = self.adaptation_tasks_brightness.get(light_id)
        color_task = self.adaptation_tasks_color.get(light_id)
        if brightness_task is not None and not brightness_task.done():
            _LOGGER.debug(
                "Cancelled ongoing brightness adaptation calls (%s) for '%s'",
                brightness_task,
                light_id,
            )
            brightness_task.cancel()
        if color_task is not None and not color_task.done():
            _LOGGER.debug(
                "Cancelled ongoing color adaptation calls (%s) for '%s'",
                color_task,
                light_id,
            )
            # color_task might be the same as brightness_task
            color_task.cancel()

    def reset(self, *lights: str, reset_manual_control: bool = True) -> None:
        """Reset the 'manual_control' status of the lights."""
        for light in lights:
            if reset_manual_control:
                _LOGGER.debug(
                    "Light %s: Clearing manual control attributes.",
                    light,
                )
                self.manual_control[light] = LightControlAttributes.NONE
                if timer := self.auto_reset_manual_control_timers.pop(light, None):
                    timer.cancel()
            self.our_last_state_on_change.pop(light, None)
            self.last_service_data.pop(light, None)
            self.cancel_ongoing_adaptation_calls(light)

    def _get_entity_list(self, service_data: ServiceData) -> list[str]:
        if ATTR_ENTITY_ID in service_data:
            return cv.ensure_list_csv(service_data[ATTR_ENTITY_ID])
        if ATTR_AREA_ID in service_data:
            entity_ids: list[str] = []
            area_ids: list[str] = cv.ensure_list_csv(service_data[ATTR_AREA_ID])
            for area_id in area_ids:
                area_entity_ids = area_entities(self.hass, area_id)
                eids = [
                    entity_id
                    for entity_id in area_entity_ids
                    if entity_id.startswith(LIGHT_DOMAIN)
                ]
                entity_ids.extend(eids)
                _LOGGER.debug(
                    "Found entity_ids '%s' for area_id '%s'",
                    entity_ids,
                    area_id,
                )
            return entity_ids
        _LOGGER.debug(
            "No entity_ids or area_ids found in service_data: %s",
            service_data,
        )
        return []

    async def turn_on_off_event_listener(self, event: Event) -> None:
        """Track 'light.turn_off' and 'light.turn_on' service calls."""
        domain = event.data.get(ATTR_DOMAIN)
        if domain != LIGHT_DOMAIN:
            return

        service = event.data[ATTR_SERVICE]
        service_data = event.data[ATTR_SERVICE_DATA]
        entity_ids = self._get_entity_list(service_data)

        if not any(eid in self.lights for eid in entity_ids):
            return

        def off(eid: str, event: Event) -> None:
            self.turn_off_event[eid] = event
            self.reset(eid)

        async def on(eid: str, event: Event) -> None:
            task = self.sleep_tasks.get(eid)
            if task is not None:
                task.cancel()
            self.turn_on_event[eid] = event

            # Only check for manual control via this path if the light was already ON.
            # Turning on from OFF is handled separately in _respond_to_off_to_on_event,
            # where adapt_only_on_bare_turn_on can mark lights as manually controlled.
            # Fix for https://github.com/basnijholt/adaptive-lighting/issues/1378
            state = self.hass.states.get(eid)
            if state is not None and state.state == STATE_ON:
                try:
                    switch = _switch_with_lights(
                        self.hass,
                        [eid],
                        expand_light_groups=False,
                    )
                    await self.update_manually_controlled_from_event(
                        switch,
                        eid,
                        force=False,
                    )
                except NoSwitchFoundError:
                    _LOGGER.debug(
                        "No switch found for entity_id='%s' in 'on' event listener",
                        eid,
                    )

            timer = self.auto_reset_manual_control_timers.get(eid)
            if (
                timer is not None
                and timer.is_running()
                and event.time_fired > timer.start_time  # type: ignore[operator]
            ):
                # Restart the auto reset timer
                timer.start()

        if service == SERVICE_TURN_OFF:
            transition = service_data.get(ATTR_TRANSITION)
            _LOGGER.debug(
                "Detected an 'light.turn_off('%s', transition=%s)' event with context.id='%s'",
                entity_ids,
                transition,
                event.context.id,
            )
            for eid in entity_ids:
                off(eid, event)

        elif service == SERVICE_TURN_ON:
            _LOGGER.debug(
                "Detected an 'light.turn_on('%s')' event with context.id='%s'",
                entity_ids,
                event.context.id,
            )
            for eid in entity_ids:
                await on(eid, event)

        elif service == SERVICE_TOGGLE:
            _LOGGER.debug(
                "Detected an 'light.toggle('%s')' event with context.id='%s'",
                entity_ids,
                event.context.id,
            )
            for eid in entity_ids:
                state = self.hass.states.get(eid)
                assert state
                self.toggle_event[eid] = event
                if state.state == STATE_ON:  # is turning off
                    off(eid, event)
                elif state.state == STATE_OFF:  # is turning on
                    await on(eid, event)

    async def state_changed_event_listener(
        self,
        event: Event[EventStateChangedData],
    ) -> None:
        """Track 'state_changed' events."""
        entity_id = event.data.get(ATTR_ENTITY_ID, "")
        if entity_id not in self.lights:
            return

        old_state = event.data.get("old_state")
        new_state = event.data.get("new_state")

        new_on = (
            new_state if new_state is not None and new_state.state == STATE_ON else None
        )
        new_off = (
            new_state
            if new_state is not None and new_state.state == STATE_OFF
            else None
        )
        old_on = (
            old_state if old_state is not None and old_state.state == STATE_ON else None
        )
        old_off = (
            old_state
            if old_state is not None and old_state.state == STATE_OFF
            else None
        )

        if new_on:
            _LOGGER.debug(
                "Detected a '%s' 'state_changed' event: '%s' with context.id='%s'",
                entity_id,
                new_on.attributes,
                new_on.context.id,
            )
            # It is possible to have multiple state change events with the same context.
            # This can happen because a `turn_on.light(brightness_pct=100, transition=30)`
            # event leads to an instant state change of
            # `new_state=dict(brightness=100, ...)`. However, after polling the light
            # could still only be `new_state=dict(brightness=50, ...)`.
            # We save all events because the first event change might indicate at what
            # settings the light will be later *or* the second event might indicate a
            # final state. The latter case happens for example when a light was
            # called with a color_temp outside of its range (and HA reports the
            # incorrect 'min_kelvin' and 'max_kelvin', which happens e.g., for
            # Philips Hue White GU10 Bluetooth lights).
            last_state: list[State] | None = self.our_last_state_on_change.get(
                entity_id,
            )
            if is_our_context(new_on.context):
                if (
                    last_state is not None
                    and last_state[0].context.id == new_on.context.id
                ):
                    _LOGGER.debug(
                        "AdaptiveLightingManager: State change event of '%s' is already"
                        " in 'self.our_last_state_on_change' (%s)"
                        " adding this state also",
                        entity_id,
                        new_on.context.id,
                    )
                    self.our_last_state_on_change[entity_id].append(new_on)
                else:
                    _LOGGER.debug(
                        "AdaptiveLightingManager: New adapt '%s' found for %s",
                        new_on,
                        entity_id,
                    )
                    self.our_last_state_on_change[entity_id] = [new_on]
                    self.start_transition_timer(entity_id)
            elif last_state is not None:
                self.our_last_state_on_change[entity_id].append(new_on)

        if old_on and new_off:
            # Tracks 'on' → 'off' state changes
            self.on_to_off_event[entity_id] = event
            self.reset(entity_id)
            _LOGGER.debug(
                "Detected an 'on' → 'off' event for '%s' with context.id='%s'",
                entity_id,
                event.context.id,
            )
        elif old_off and new_on:
            # Tracks 'off' → 'on' state changes
            self.off_to_on_event[entity_id] = event
            _LOGGER.debug(
                "Detected an 'off' → 'on' event for '%s' with context.id='%s'",
                entity_id,
                event.context.id,
            )

            if self.is_proactively_adapting(event.context.id):
                _LOGGER.debug(
                    "Skipping responding to 'off' → 'on' event for '%s' with context.id='%s' because"
                    " we are already proactively adapting",
                    entity_id,
                    event.context.id,
                )
                # Note: the reset below already happened in `_service_interceptor_turn_on_handler`
                return

            self.reset(entity_id, reset_manual_control=False)
            lock = self.turn_off_locks.setdefault(entity_id, asyncio.Lock())
            async with lock:
                if await self.just_turned_off(entity_id):
                    # Stop if a rapid 'off' → 'on' → 'off' happens.
                    _LOGGER.debug(
                        "Cancelling adjusting lights for %s",
                        entity_id,
                    )
                    return

            switches = _switches_with_lights(self.hass, [entity_id])
            for switch in switches:
                if switch.is_on:
                    await switch._respond_to_off_to_on_event(
                        entity_id,
                        event,
                    )

    async def update_manually_controlled_from_event(
        self,
        switch: AdaptiveSwitch,
        light: str,
        force: bool,
    ) -> None:
        """Check if the light has been manually controlled by the latest turn on event."""
        if not switch._take_over_control:
            return

        turn_on_event = self.turn_on_event.get(light)

        if (
            turn_on_event is None
            or self.is_proactively_adapting(turn_on_event.context.id)
            or is_our_context(turn_on_event.context)
            or force
        ):
            return

        turn_on_attributes = get_light_control_attributes(
            turn_on_event.data[ATTR_SERVICE_DATA],
        )

        if not turn_on_attributes:
            return

        # Light was already on and 'light.turn_on' was not called by
        # the adaptive_lighting integration.
        self.add_manual_control_attributes(light, turn_on_attributes)
        switch.fire_manual_control_event(light, turn_on_event.context)
        _LOGGER.debug(
            "'%s' was already on and 'light.turn_on' was not called by the"
            " adaptive_lighting integration (context.id='%s'), the Adaptive"
            " Lighting will stop adapting %s of the light until the switch or the"
            " light turns off and then on again.",
            light,
            turn_on_event.context.id,
            turn_on_attributes,
        )

    async def update_manually_controlled_from_untracked_change(
        self,
        switch: AdaptiveSwitch,
        light: str,
        force: bool,
        context: Context,
    ) -> None:
        """Check if the light has been manually controlled from an untracked change.

        An untracked change is a change that has been made outsideof HA and is
        therefore not visible through events.
        """
        if not switch._take_over_control or not switch._detect_non_ha_changes or force:
            return

        # Note: This call updates the state of the light
        # so it might suddenly be off.
        significantly_changed_attributes = await self.significant_change(
            switch,
            light,
            context,
        )

        if not significantly_changed_attributes:
            return

        self.add_manual_control_attributes(
            light,
            significantly_changed_attributes,
        )
        switch.fire_manual_control_event(light, context)

    async def significant_change(
        self,
        switch: AdaptiveSwitch,
        light: str,
        context: Context,  # just for logging
    ) -> LightControlAttributes:
        """Has the light made a significant change since last update.

        This method will detect changes that were made to the light without
        calling 'light.turn_on', so outside of Home Assistant.
        """
        assert switch._detect_non_ha_changes

        last_service_data = self.last_service_data.get(light)
        if last_service_data is None:
            return LightControlAttributes.NONE
        # Update state and check for a manual change not done in HA.
        # Ensure HASS is correctly updating your light's state with
        # light.turn_on calls if any problems arise. This
        # can happen e.g. using zigbee2mqtt with 'report: false' in device settings.
        await async_update_entity(self.hass, light)
        refreshed_state = self.hass.states.get(light)
        assert refreshed_state is not None

        changed_attributes = _attributes_have_changed(
            old_attributes=last_service_data,
            new_attributes=refreshed_state.attributes,
            light=light,
            context=context,
        )
        if changed_attributes:
            _LOGGER.debug(
                "%s: State attributes %s of '%s' changed (%s) wrt 'last_service_data' (%s) (context.id=%s)",
                switch._name,
                changed_attributes,
                light,
                refreshed_state.attributes,
                last_service_data,
                context.id,
            )
        else:
            _LOGGER.debug(
                "%s: State attributes of '%s' did not change (%s) wrt 'last_service_data' (%s) (context.id=%s)",
                switch._name,
                light,
                refreshed_state.attributes,
                last_service_data,
                context.id,
            )
        return changed_attributes

    def _off_to_on_state_event_is_from_turn_on(
        self,
        entity_id: str,
        off_to_on_event: Event[EventStateChangedData],
    ) -> bool:
        # Adaptive Lighting should never turn on lights itself
        if is_our_context(off_to_on_event.context) and not is_our_context(
            off_to_on_event.context,
            "service",  # adaptive_lighting.apply is allowed to turn on lights
        ):
            _LOGGER.warning(
                "Detected an 'off' → 'on' event for '%s' with context.id='%s',"
                " triggered by the adaptive_lighting integration itself,"
                " which *should* not happen. If you see this please submit an issue with"
                " your full logs at https://github.com/basnijholt/adaptive-lighting",
                entity_id,
                off_to_on_event.context.id,
            )
            _LOGGER.debug(
                "Full 'off' → 'on' event for '%s': %s",
                entity_id,
                off_to_on_event,
            )
        turn_on_event: Event | None = self.turn_on_event.get(entity_id)
        id_off_to_on = off_to_on_event.context.id
        return turn_on_event is not None and id_off_to_on == turn_on_event.context.id

    def _member_turn_on_explains_group_turn_on(
        self,
        entity_id: str,
        on_to_off_event: Event[EventStateChangedData],
        off_to_on_event: Event[EventStateChangedData],
    ) -> bool:
        """Check if a light group's 'off' → 'on' is caused by a member's 'light.turn_on'.

        When a member of a light group is turned on while the group is off, the
        group turns on as a side effect. Home Assistant may reuse the context of
        an earlier 'light.turn_off' call for the group's state change (entities
        keep their context for a few seconds), which makes the group's turn-on
        look like a polling artifact of the turn-off.
        See https://github.com/basnijholt/adaptive-lighting/issues/1378
        """
        state = self.hass.states.get(entity_id)
        if state is None or not _is_light_group(state):
            return False
        members: list[str] = state.attributes[ATTR_ENTITY_ID]
        for member in members:
            member_turn_on = self.turn_on_event.get(member)
            if (
                member_turn_on is not None
                and on_to_off_event.time_fired
                < member_turn_on.time_fired
                <= off_to_on_event.time_fired
            ):
                _LOGGER.debug(
                    "just_turned_off: Light group '%s' turned on because its member"
                    " '%s' was turned on (context.id='%s'), so this is a legitimate"
                    " turn-on, not a polling artifact.",
                    entity_id,
                    member,
                    member_turn_on.context.id,
                )
                return True
        return False

    async def just_turned_off(  # noqa: PLR0911, PLR0912
        self,
        entity_id: str,
    ) -> bool:
        """Cancel the adjusting of a light if it has just been turned off.

        Possibly the lights just got a 'turn_off' call, however, the light
        is actually still turning off (e.g., because of a 'transition') and
        HA polls the light before the light is 100% off. This might trigger
        a rapid switch 'off' → 'on' → 'off'. To prevent this component
        from interfering on the 'on' state, we make sure to wait at least
        TURNING_OFF_DELAY (or the 'turn_off' transition time) between a
        'off' → 'on' event and then check whether the light is still 'on' or
        if the brightness is still decreasing. Only if it is the case we
        adjust the lights.
        """
        off_to_on_event = self.off_to_on_event[entity_id]
        on_to_off_event = self.on_to_off_event.get(entity_id)

        if on_to_off_event is None:
            _LOGGER.debug(
                "just_turned_off: No 'on' → 'off' state change has been registered before for '%s'."
                " It's possible that the light was already on when Home Assistant was turned on.",
                entity_id,
            )
            return False

        if off_to_on_event.context.id == on_to_off_event.context.id:
            # Matching context IDs usually mean a polling artifact (HA briefly
            # reports 'on' while the light is still turning off). However, the
            # context is also reused when e.g. one automation turns the light
            # off and later back on, or when an integration writes the state
            # with the entity's cached context. Only treat the state change as
            # a legitimate turn-on if a 'light.turn_on' call for this light (or
            # for a member of this light group) fired between the two state
            # changes.
            turn_on_event = self.turn_on_event.get(entity_id)
            if (
                turn_on_event is not None
                and on_to_off_event.time_fired
                < turn_on_event.time_fired
                <= off_to_on_event.time_fired
            ):
                _LOGGER.debug(
                    "just_turned_off: 'light.turn_on' was called for '%s' between its"
                    " 'on' → 'off' and 'off' → 'on' state changes, so this is a"
                    " legitimate turn-on, not a polling artifact.",
                    entity_id,
                )
                return False
            if self._member_turn_on_explains_group_turn_on(
                entity_id,
                on_to_off_event,
                off_to_on_event,
            ):
                return False
            _LOGGER.debug(
                "just_turned_off: 'on' → 'off' state change has the same context.id as the"
                " 'off' → 'on' state change for '%s'. This is probably a false positive.",
                entity_id,
            )
            return True

        id_on_to_off = on_to_off_event.context.id

        turn_off_event = self.turn_off_event.get(entity_id)
        if turn_off_event is not None:
            transition = turn_off_event.data[ATTR_SERVICE_DATA].get(ATTR_TRANSITION)
        else:
            transition = None

        if self._off_to_on_state_event_is_from_turn_on(entity_id, off_to_on_event):
            is_toggle = off_to_on_event == self.toggle_event.get(entity_id)
            from_service = "light.toggle" if is_toggle else "light.turn_on"
            _LOGGER.debug(
                "just_turned_off: State change 'off' → 'on' triggered by '%s'",
                from_service,
            )
            return False

        if (
            turn_off_event is not None
            and id_on_to_off == turn_off_event.context.id
            and transition is not None  # 'turn_off' is called with transition=...
        ):
            # State change 'on' → 'off' and 'light.turn_off(..., transition=...)' come
            # from the same event, so wait at least the 'turn_off' transition time.
            delay = max(transition, TURNING_OFF_DELAY)
        else:
            # State change 'off' → 'on' happened because the light state was set.
            # Possibly because of polling.
            delay = TURNING_OFF_DELAY

        delta_time = (dt_util.utcnow() - on_to_off_event.time_fired).total_seconds()
        if delta_time > delay:
            _LOGGER.debug(
                "just_turned_off: delta_time='%s' > delay='%s'",
                delta_time,
                delay,
            )
            return False

        # Here we could just `return True` but because we want to prevent any updates
        # from happening to this light (through async_track_time_interval or
        # sleep_state) for some time, we wait below until the light
        # is 'off' or the time has passed.

        delay -= delta_time  # delta_time has passed since the 'off' → 'on' event
        _LOGGER.debug(
            "just_turned_off: Waiting with adjusting '%s' for %s",
            entity_id,
            delay,
        )
        total_sleep = 0
        for _ in range(3):
            # It can happen that the actual transition time is longer than the
            # specified time in the 'turn_off' service.
            coro = asyncio.sleep(delay)
            total_sleep += delay
            task = self.sleep_tasks[entity_id] = asyncio.ensure_future(coro)
            try:
                await task
            except asyncio.CancelledError:  # 'light.turn_on' has been called
                _LOGGER.debug(
                    "just_turned_off: Sleep task is cancelled due to 'light.turn_on('%s')' call",
                    entity_id,
                )
                return False

            if not is_on(self.hass, entity_id):
                _LOGGER.debug(
                    "just_turned_off: '%s' is off after %s seconds, cancelling adaptation",
                    entity_id,
                    total_sleep,
                )
                return True
            delay = TURNING_OFF_DELAY  # next time only wait this long

        if transition is not None:
            # Always ignore when there's a 'turn_off' transition.
            # Because it seems like HA cannot detect whether a light is
            # transitioning into 'off'. Maybe needs some discussion/input?
            return True

        # Now we assume that the lights are still on and they were intended
        # to be on.
        _LOGGER.debug(
            "just_turned_off: '%s' is still on after %s seconds, assuming it was intended to be on",
            entity_id,
            total_sleep,
        )
        return False

    def _mark_manual_control_if_non_bare_turn_on(
        self,
        entity_id: str,
        service_data: ServiceData,
    ) -> bool:
        """Mark light as manually controlled if turn_on call has brightness/color attributes.

        This is used by adapt_only_on_bare_turn_on to mark lights as manually controlled
        when they are turned on with specific attributes (e.g., from a scene).
        This ensures scenes persist and AL doesn't override them.
        """
        _LOGGER.debug(
            "_mark_manual_control_if_non_bare_turn_on: entity_id='%s', service_data='%s'",
            entity_id,
            service_data,
        )
        manual_control_attributes = get_light_control_attributes(service_data)

        if manual_control_attributes:
            self.set_manual_control_attributes(entity_id, manual_control_attributes)
            return True

        return False


class _AsyncSingleShotTimer:
    def __init__(self, delay: float, callback: Callable[[], None | Any]) -> None:
        """Initialize the timer."""
        self.delay = delay
        self.callback = callback
        self.task = None
        self.start_time: datetime.datetime | None = None

    async def _run(self) -> None:
        """Run the timer. Don't call this directly, use start() instead."""
        await asyncio.sleep(self.delay)
        if self.callback:
            if asyncio.iscoroutinefunction(self.callback):
                await self.callback()
            else:
                self.callback()

    def is_running(self) -> bool:
        """Return whether the timer is running."""
        return self.task is not None and not self.task.done()

    def start(self) -> None:
        """Start the timer."""
        if self.task is not None and not self.task.done():
            self.task.cancel()
        # Set start_time before creating task to avoid race condition
        # where is_running() returns True but start_time is still None
        # See: https://github.com/basnijholt/adaptive-lighting/issues/1272
        self.start_time = dt_util.utcnow()
        self.task = asyncio.create_task(self._run())

    def cancel(self) -> None:
        """Cancel the timer."""
        if self.task:
            self.task.cancel()
            self.callback = None

    def remaining_time(self) -> float:
        """Return the remaining time before the timer expires."""
        if self.start_time is not None:
            elapsed_time = (dt_util.utcnow() - self.start_time).total_seconds()
            return max(0, self.delay - elapsed_time)
        return 0
