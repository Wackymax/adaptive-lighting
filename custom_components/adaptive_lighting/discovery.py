"""Discover and classify Home Assistant entities for Adaptive Lighting.

Discovery is deliberately an observation layer.  It reports what is present in
the configured areas, but it does not create a service-call client or grant any
entity actuation authority.  An owner must still explicitly configure the
entities it is allowed to control.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, fields
from typing import Any

from homeassistant.components.binary_sensor import BinarySensorDeviceClass
from homeassistant.components.cover import CoverDeviceClass
from homeassistant.components.light import (
    ATTR_SUPPORTED_COLOR_MODES,
    ColorMode,
)
from homeassistant.const import (
    ATTR_DEVICE_CLASS,
    ATTR_FRIENDLY_NAME,
    ATTR_UNIT_OF_MEASUREMENT,
    STATE_UNAVAILABLE,
    STATE_UNKNOWN,
)
from homeassistant.core import Event, HomeAssistant, State, callback
from homeassistant.helpers import area_registry, device_registry, entity_registry

_LOGGER = logging.getLogger(__name__)

LIGHT_DOMAIN = "light"
SENSOR_DOMAIN = "sensor"
BINARY_SENSOR_DOMAIN = "binary_sensor"
COVER_DOMAIN = "cover"
MEDIA_PLAYER_DOMAIN = "media_player"
ALARM_CONTROL_PANEL_DOMAIN = "alarm_control_panel"
CALENDAR_DOMAIN = "calendar"
WEATHER_DOMAIN = "weather"
SUN_DOMAIN = "sun"
PERSON_DOMAIN = "person"

STATUS_AVAILABLE = "available"
STATUS_UNAVAILABLE = "unavailable"
STATUS_MISSING = "missing"
STATUS_DISABLED = "disabled"
STATUS_HIDDEN = "hidden"
STATUS_UNASSIGNED = "unassigned"

_CONTEXT_CAPABILITIES = frozenset(
    {
        "motion",
        "occupancy",
        "presence",
        "opening",
        "door",
        "window",
        "garage_door",
        "illuminance",
        "media",
        "security",
        "holiday_calendar",
        "weather",
        "daylight_proxy",
        "household_presence",
        "arrival",
    },
)


def _value(value: Any) -> str | None:
    """Return a stable string for a HA enum or string value."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    enum_value = getattr(value, "value", None)
    return enum_value if isinstance(enum_value, str) else str(value)


def _normalise_ids(entity_ids: Iterable[str]) -> tuple[str, ...]:
    """Return deterministic, non-empty entity IDs."""
    return tuple(sorted({entity_id for entity_id in entity_ids if entity_id}))


def _deduplicate(values: Iterable[str]) -> tuple[str, ...]:
    """Deduplicate capability names without changing their semantic order."""
    return tuple(dict.fromkeys(values))


def _supported_color_modes(
    entry: entity_registry.RegistryEntry,
    state: State | None,
) -> tuple[str, ...]:
    """Read light modes from live state, falling back to registry capabilities."""
    values: Any = None
    if state is not None:
        values = state.attributes.get(ATTR_SUPPORTED_COLOR_MODES)
    if values is None and entry.capabilities:
        values = entry.capabilities.get(ATTR_SUPPORTED_COLOR_MODES)
    if isinstance(values, str):
        values = (values,)
    if not isinstance(values, Iterable):
        return ()
    return tuple(sorted({_value(value) for value in values if _value(value)}))


@dataclass(frozen=True, slots=True)
class EntityClassification:
    """Pure classification output for one registry entry."""

    capabilities: tuple[str, ...]
    supported_color_modes: tuple[str, ...] = ()
    dimmable: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe classification data."""
        return {
            "capabilities": list(self.capabilities),
            "supported_color_modes": list(self.supported_color_modes),
            "dimmable": self.dimmable,
        }


def classify_entity(  # noqa: PLR0912,PLR0915 - one auditable branch per HA domain
    entry: entity_registry.RegistryEntry,
    state: State | None = None,
) -> EntityClassification:
    """Classify an entity using registry metadata and, when present, live state.

    The domain remains authoritative for whether something is a light.  In
    particular, a switch that happens to drive a lamp is not promoted to a
    controllable light.  Light dimmability comes from the HA supported color
    mode contract, with the registry capabilities used when state is absent.
    """
    domain = entry.domain
    device_class = _value(
        (state.attributes.get(ATTR_DEVICE_CLASS) if state is not None else None)
        or entry.device_class,
    )
    capabilities: list[str] = []
    supported_color_modes: tuple[str, ...] = ()
    dimmable = False

    if domain == LIGHT_DOMAIN:
        supported_color_modes = _supported_color_modes(entry, state)
        dimmable = any(
            mode in {mode.value for mode in ColorMode if mode != ColorMode.ONOFF}
            for mode in supported_color_modes
        )
        # A live brightness attribute is a useful compatibility fallback for a
        # registry entry created before the integration exposed capabilities.
        if not dimmable and state is not None:
            dimmable = state.attributes.get("brightness") is not None
        capabilities.extend(("light", "on_off_light"))
        if dimmable:
            capabilities.append("dimmable_light")
    elif domain == BINARY_SENSOR_DOMAIN:
        capabilities.append("binary_sensor")
        binary_classes = {
            BinarySensorDeviceClass.MOTION.value: ("motion",),
            BinarySensorDeviceClass.OCCUPANCY.value: ("occupancy",),
            BinarySensorDeviceClass.PRESENCE.value: ("presence",),
            BinarySensorDeviceClass.OPENING.value: ("opening",),
            BinarySensorDeviceClass.DOOR.value: ("opening", "door"),
            BinarySensorDeviceClass.WINDOW.value: ("opening", "window"),
            BinarySensorDeviceClass.GARAGE_DOOR.value: ("opening", "garage_door"),
        }
        if device_class in binary_classes:
            capabilities.extend(binary_classes[device_class])
        entity_name = entry.entity_id.lower()
        if "arrival" in entity_name or "just_arrived" in entity_name:
            capabilities.extend(("arrival", "context"))
        if any(token in entity_name for token in ("household_home", "house_occupied")):
            capabilities.extend(("household_presence", "context"))
        # Home Assistant's Workday integration can be configured as a precise
        # public-holiday sensor by including only ``holiday``.  Detect that
        # semantic contract from state attributes instead of relying on a
        # user-selected entity name.
        if state is not None and set(state.attributes.get("workdays", ())) == {
            "holiday",
        }:
            capabilities.extend(("holiday_calendar", "context"))
    elif domain == COVER_DOMAIN:
        capabilities.append("cover")
        cover_classes = {
            CoverDeviceClass.DOOR.value: ("opening", "door"),
            CoverDeviceClass.WINDOW.value: ("opening", "window"),
            CoverDeviceClass.GARAGE.value: ("opening", "garage_door"),
        }
        capabilities.extend(cover_classes.get(device_class, ()))
    elif domain == SENSOR_DOMAIN:
        capabilities.append("sensor")
        unit = (
            state.attributes.get(ATTR_UNIT_OF_MEASUREMENT)
            if state is not None
            else None
        ) or entry.unit_of_measurement
        if device_class == "illuminance" or str(unit).lower() in {"lx", "lux"}:
            capabilities.extend(("illuminance", "context"))
        sensor_name = " ".join(
            value
            for value in (entry.entity_id, entry.name, entry.original_name)
            if value
        ).lower()
        if any(
            token in sensor_name
            for token in ("solar", "pv", "photovoltaic", "inverter")
        ):
            capabilities.extend(("daylight_proxy", "solar_proxy", "context"))
    elif domain == MEDIA_PLAYER_DOMAIN:
        capabilities.extend(("media_player", "media", "context"))
    elif domain == ALARM_CONTROL_PANEL_DOMAIN:
        capabilities.extend(("alarm_control_panel", "security", "context"))
    elif domain == CALENDAR_DOMAIN:
        name = " ".join(
            value
            for value in (
                entry.entity_id,
                entry.name,
                entry.original_name,
                state.attributes.get(ATTR_FRIENDLY_NAME) if state else None,
            )
            if value
        ).lower()
        capabilities.append("calendar")
        if any(token in name for token in ("holiday", "public holiday", "holidays")):
            capabilities.extend(("holiday_calendar", "context"))
    elif domain == WEATHER_DOMAIN:
        capabilities.extend(("weather", "daylight_proxy", "context"))
    elif domain == SUN_DOMAIN:
        capabilities.extend(("sun", "daylight_proxy", "context"))
    elif domain == PERSON_DOMAIN:
        capabilities.extend(("person", "household_presence", "context"))
    else:
        capabilities.append(
            domain if domain in {"switch", "fan", "climate"} else "other",
        )

    if (
        any(capability in _CONTEXT_CAPABILITIES for capability in capabilities)
        and "context" not in capabilities
    ):
        capabilities.append("context")

    return EntityClassification(
        capabilities=_deduplicate(capabilities),
        supported_color_modes=supported_color_modes,
        dimmable=dimmable,
    )


def effective_area_id(
    entry: entity_registry.RegistryEntry,
    device: device_registry.DeviceEntry | None,
) -> str | None:
    """Resolve area using the entity override before the device assignment.

    HA permits an entity to override its device area.  Treating the device area
    as authoritative would silently move such an entity into another control
    cohort when a shared device is reassigned.
    """
    if entry.area_id is not None:
        return entry.area_id
    return device.area_id if device is not None else None


def _state_status(state: State | None) -> tuple[str, bool]:
    """Return diagnostic availability without treating missing as available."""
    if state is None:
        return STATUS_MISSING, False
    if state.state in {STATE_UNAVAILABLE, STATE_UNKNOWN}:
        return STATUS_UNAVAILABLE, False
    return STATUS_AVAILABLE, True


@dataclass(frozen=True, slots=True)
class EntityInventory:
    """Immutable, JSON-safe description of one discovered entity."""

    entity_id: str
    domain: str
    name: str
    area_id: str | None
    area_name: str | None
    device_id: str | None
    device_name: str | None
    device_class: str | None
    capabilities: tuple[str, ...]
    actionable_capabilities: tuple[str, ...]
    supported_color_modes: tuple[str, ...]
    status: str
    available: bool
    explicit_controlled: bool
    discovered_candidate: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""
        return {
            "entity_id": self.entity_id,
            "domain": self.domain,
            "name": self.name,
            "area_id": self.area_id,
            "area_name": self.area_name,
            "device_id": self.device_id,
            "device_name": self.device_name,
            "device_class": self.device_class,
            "capabilities": list(self.capabilities),
            "actionable_capabilities": list(self.actionable_capabilities),
            "supported_color_modes": list(self.supported_color_modes),
            "status": self.status,
            "available": self.available,
            "explicit_controlled": self.explicit_controlled,
            "discovered_candidate": self.discovered_candidate,
        }


@dataclass(frozen=True, slots=True)
class InventoryMove:
    """Area move delta for an existing entity."""

    entity_id: str
    old_area_id: str | None
    new_area_id: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""
        return {
            "entity_id": self.entity_id,
            "old_area_id": self.old_area_id,
            "new_area_id": self.new_area_id,
        }


@dataclass(frozen=True, slots=True)
class InventoryChange:
    """Non-area change delta for an existing entity."""

    entity_id: str
    old: EntityInventory
    new: EntityInventory
    fields: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""
        return {
            "entity_id": self.entity_id,
            "old": self.old.to_dict(),
            "new": self.new.to_dict(),
            "fields": list(self.fields),
        }


@dataclass(frozen=True, slots=True)
class InventoryRename:
    """Entity ID rename delta from an entity registry update."""

    old_entity_id: str
    new_entity_id: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-safe representation."""
        return {
            "old_entity_id": self.old_entity_id,
            "new_entity_id": self.new_entity_id,
        }


@dataclass(frozen=True, slots=True)
class DiscoveryDiagnostic:
    """Diagnostic for a configured seed that cannot join an area inventory."""

    entity_id: str
    status: str
    reason: str
    area_id: str | None
    explicit_controlled: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""
        return {
            "entity_id": self.entity_id,
            "status": self.status,
            "reason": self.reason,
            "area_id": self.area_id,
            "explicit_controlled": self.explicit_controlled,
        }


@dataclass(frozen=True, slots=True)
class InventorySnapshot:
    """Immutable inventory publication and its reconciliation deltas."""

    revision: int
    reason: str
    monitored_area_ids: tuple[str, ...]
    entities: tuple[EntityInventory, ...]
    diagnostics: tuple[DiscoveryDiagnostic, ...]
    added: tuple[EntityInventory, ...] = ()
    removed: tuple[EntityInventory, ...] = ()
    moved: tuple[InventoryMove, ...] = ()
    changed: tuple[InventoryChange, ...] = ()
    renamed: tuple[InventoryRename, ...] = ()

    @property
    def inventory(self) -> tuple[EntityInventory, ...]:
        """Alias for callers that use inventory terminology."""
        return self.entities

    @property
    def items(self) -> tuple[EntityInventory, ...]:
        """Alias for callers that use item terminology."""
        return self.entities

    @property
    def entity_ids(self) -> tuple[str, ...]:
        """Return entity IDs in deterministic order."""
        return tuple(item.entity_id for item in self.entities)

    def to_dict(self) -> dict[str, Any]:
        """Return a complete JSON-safe snapshot."""
        return {
            "revision": self.revision,
            "reason": self.reason,
            "monitored_area_ids": list(self.monitored_area_ids),
            "entities": [item.to_dict() for item in self.entities],
            "diagnostics": [item.to_dict() for item in self.diagnostics],
            "added": [item.to_dict() for item in self.added],
            "removed": [item.to_dict() for item in self.removed],
            "moved": [item.to_dict() for item in self.moved],
            "changed": [item.to_dict() for item in self.changed],
            "renamed": [item.to_dict() for item in self.renamed],
        }


ChangeCallback = Callable[[InventorySnapshot], Awaitable[None] | None]


class EntityDiscoveryCoordinator:
    """Maintain a live, area-scoped entity inventory."""

    def __init__(
        self,
        hass: HomeAssistant,
        *,
        seed_entity_ids: Iterable[str] = (),
        light_entity_ids: Iterable[str] = (),
        context_entity_ids: Iterable[str] = (),
        explicit_controlled_entity_ids: Iterable[str] = (),
        debounce_seconds: float = 0.05,
        on_change: ChangeCallback | None = None,
    ) -> None:
        """Create a coordinator with explicit seed and control boundaries."""
        self.hass = hass
        self.seed_entity_ids = _normalise_ids(
            (*seed_entity_ids, *light_entity_ids, *context_entity_ids),
        )
        self.explicit_controlled_entity_ids = frozenset(
            _normalise_ids(explicit_controlled_entity_ids),
        )
        self.debounce_seconds = max(0.0, float(debounce_seconds))
        self._on_change = on_change
        self._entity_registry = entity_registry.async_get(hass)
        try:
            self._device_registry = device_registry.async_get(hass)
        except RuntimeError:
            # Device registry setup is normally complete before integrations
            # start.  Keeping this nullable makes pure/read-only use safe while
            # still using the exact HA registry when it is available.
            self._device_registry = None
        self._area_registry = area_registry.async_get(hass)
        self._unsubscribers: list[Callable[[], None]] = []
        self._refresh_task: asyncio.Task[None] | None = None
        self._refresh_lock = asyncio.Lock()
        self._pending_reasons: set[str] = set()
        self._pending_renames: set[tuple[str, str]] = set()
        self._started = False
        self._has_published = False
        self._snapshot = InventorySnapshot(
            revision=0,
            reason="not_started",
            monitored_area_ids=(),
            entities=(),
            diagnostics=(),
        )

    @property
    def snapshot(self) -> InventorySnapshot:
        """Return the most recently published snapshot."""
        return self._snapshot

    @property
    def current_snapshot(self) -> InventorySnapshot:
        """Alias for :attr:`snapshot`."""
        return self._snapshot

    @property
    def started(self) -> bool:
        """Whether registry listeners are installed."""
        return self._started

    async def async_start(self) -> InventorySnapshot:
        """Install listeners and publish the initial inventory."""
        if self._started:
            return self._snapshot
        self._started = True
        self._unsubscribers = [
            self.hass.bus.async_listen(
                entity_registry.EVENT_ENTITY_REGISTRY_UPDATED,
                self._handle_entity_registry_event,
            ),
            self.hass.bus.async_listen(
                device_registry.EVENT_DEVICE_REGISTRY_UPDATED,
                self._handle_device_registry_event,
            ),
            self.hass.bus.async_listen(
                area_registry.EVENT_AREA_REGISTRY_UPDATED,
                self._handle_area_registry_event,
            ),
        ]
        return await self.async_refresh(reason="start")

    async def start(self) -> InventorySnapshot:
        """Public lifecycle alias for :meth:`async_start`."""
        return await self.async_start()

    async def async_stop(self) -> None:
        """Remove listeners and cancel any coalescing task."""
        if not self._started and not self._unsubscribers and self._refresh_task is None:
            return
        self._started = False
        for unsubscribe in self._unsubscribers:
            unsubscribe()
        self._unsubscribers.clear()
        self._pending_reasons.clear()
        self._pending_renames.clear()
        task = self._refresh_task
        self._refresh_task = None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def stop(self) -> None:
        """Public lifecycle alias for :meth:`async_stop`."""
        await self.async_stop()

    async def async_refresh(self, reason: str = "manual") -> InventorySnapshot:
        """Reconcile registries immediately and publish if materially changed."""
        async with self._refresh_lock:
            area_ids = self._derive_monitored_area_ids()
            entities, diagnostics = self._build_inventory(area_ids)
            previous = self._snapshot
            rename_pairs = self._consume_pending_renames()
            previous_by_id = {item.entity_id: item for item in previous.entities}
            current_by_id = {item.entity_id: item for item in entities}

            renamed = tuple(
                InventoryRename(old_entity_id=old_id, new_entity_id=new_id)
                for old_id, new_id in sorted(rename_pairs)
                if old_id in previous_by_id and new_id in current_by_id
            )
            renamed_old = {item.old_entity_id for item in renamed}
            renamed_new = {item.new_entity_id for item in renamed}
            added = tuple(
                current_by_id[entity_id]
                for entity_id in sorted(current_by_id.keys() - previous_by_id.keys())
                if entity_id not in renamed_new
            )
            removed = tuple(
                previous_by_id[entity_id]
                for entity_id in sorted(previous_by_id.keys() - current_by_id.keys())
                if entity_id not in renamed_old
            )
            moved: list[InventoryMove] = []
            changed: list[InventoryChange] = []
            comparable_fields = tuple(
                field.name
                for field in fields(EntityInventory)
                if field.name not in {"entity_id", "area_id"}
            )
            for entity_id in sorted(previous_by_id.keys() & current_by_id.keys()):
                old = previous_by_id[entity_id]
                new = current_by_id[entity_id]
                if old.area_id != new.area_id:
                    moved.append(
                        InventoryMove(
                            entity_id=entity_id,
                            old_area_id=old.area_id,
                            new_area_id=new.area_id,
                        ),
                    )
                changed_fields = tuple(
                    field_name
                    for field_name in comparable_fields
                    if getattr(old, field_name) != getattr(new, field_name)
                )
                if changed_fields:
                    changed.append(
                        InventoryChange(
                            entity_id=entity_id,
                            old=old,
                            new=new,
                            fields=changed_fields,
                        ),
                    )

            meaningful = (
                not self._has_published
                or area_ids != previous.monitored_area_ids
                or entities != previous.entities
                or diagnostics != previous.diagnostics
            )
            revision = previous.revision + 1 if meaningful else previous.revision
            snapshot = InventorySnapshot(
                revision=revision,
                reason=reason,
                monitored_area_ids=tuple(sorted(area_ids)),
                entities=entities,
                diagnostics=diagnostics,
                added=added,
                removed=removed,
                moved=tuple(moved),
                changed=tuple(changed),
                renamed=renamed,
            )
            self._snapshot = snapshot
            self._has_published = True

        if meaningful and self._on_change is not None:
            try:
                result = self._on_change(snapshot)
                if inspect.isawaitable(result):
                    await result
            except Exception:  # pragma: no cover - owner boundary
                _LOGGER.exception("Adaptive Lighting discovery callback failed")
        return snapshot

    async def refresh(self, reason: str = "manual") -> InventorySnapshot:
        """Public refresh alias for :meth:`async_refresh`."""
        return await self.async_refresh(reason)

    @callback
    def _handle_entity_registry_event(self, event: Event) -> None:
        """Coalesce entity create/remove/update/rename events."""
        data = event.data
        if data.get("action") == "update" and data.get("old_entity_id"):
            self._pending_renames.add(
                (data["old_entity_id"], data.get("entity_id", "")),
            )
        self._schedule_refresh(f"entity_registry:{data.get('action', 'update')}")

    @callback
    def _handle_device_registry_event(self, event: Event) -> None:
        """Refresh after device area/name or lifecycle changes."""
        self._schedule_refresh(f"device_registry:{event.data.get('action', 'update')}")

    @callback
    def _handle_area_registry_event(self, event: Event) -> None:
        """Refresh after an area is renamed, removed, or otherwise changed."""
        self._schedule_refresh(f"area_registry:{event.data.get('action', 'update')}")

    def _schedule_refresh(self, reason: str) -> None:
        """Schedule one tracked debounce task for a burst of registry events."""
        if not self._started:
            return
        self._pending_reasons.add(reason)
        if self._refresh_task is None or self._refresh_task.done():
            self._refresh_task = self.hass.async_create_task(
                self._async_run_debounced_refresh(),
                name="adaptive_lighting_discovery_refresh",
            )

    async def _async_run_debounced_refresh(self) -> None:
        """Debounce registry bursts without leaving an untracked task behind."""
        try:
            while self._started:
                await asyncio.sleep(self.debounce_seconds)
                if not self._started:
                    return
                reasons = tuple(sorted(self._pending_reasons))
                self._pending_reasons.clear()
                await self.async_refresh(reason=";".join(reasons) or "registry_update")
                if not self._pending_reasons:
                    return
        finally:
            if self._refresh_task is asyncio.current_task():
                self._refresh_task = None

    def _consume_pending_renames(self) -> set[tuple[str, str]]:
        """Consume rename metadata only when a refresh reconciles it."""
        renames = self._pending_renames
        self._pending_renames = set()
        return renames

    def _device_for(
        self,
        entry: entity_registry.RegistryEntry,
    ) -> device_registry.DeviceEntry | None:
        """Resolve a device without making a missing device registry fatal."""
        if self._device_registry is None or entry.device_id is None:
            return None
        return self._device_registry.async_get(entry.device_id)

    def _derive_monitored_area_ids(self) -> set[str]:
        """Derive areas only from explicit configured seed entities."""
        area_ids: set[str] = set()
        for entity_id in self.seed_entity_ids:
            entry = self._entity_registry.async_get(entity_id)
            if entry is None:
                continue
            if area_id := effective_area_id(entry, self._device_for(entry)):
                area_ids.add(area_id)
        return area_ids

    def _build_inventory(
        self,
        area_ids: set[str],
    ) -> tuple[tuple[EntityInventory, ...], tuple[DiscoveryDiagnostic, ...]]:
        """Build a complete, deterministic registry/state-machine snapshot."""
        entities: list[EntityInventory] = []
        for entry in sorted(
            self._entity_registry.entities.values(),
            key=lambda item: item.entity_id,
        ):
            device = self._device_for(entry)
            area_id = effective_area_id(entry, device)
            if entry.disabled or entry.hidden:
                continue
            state = self.hass.states.get(entry.entity_id)
            classification = classify_entity(entry, state)
            # Holiday calendars are installation-wide temporal context and
            # commonly have no area. Other entities remain area-scoped.
            is_global_context = bool(
                {
                    "holiday_calendar",
                    "weather",
                    "daylight_proxy",
                    "household_presence",
                }
                & set(classification.capabilities),
            )
            if area_id not in area_ids and not is_global_context:
                continue
            status, available = _state_status(state)
            area = self._area_registry.async_get_area(area_id) if area_id else None
            name = (
                entry.name
                or entry.original_name_unprefixed
                or entry.original_name
                or (state.attributes.get(ATTR_FRIENDLY_NAME) if state else None)
                or entry.entity_id
            )
            capabilities = classification.capabilities
            entities.append(
                EntityInventory(
                    entity_id=entry.entity_id,
                    domain=entry.domain,
                    name=name,
                    area_id=area_id,
                    area_name=area.name if area else None,
                    device_id=entry.device_id,
                    device_name=device.name if device else None,
                    device_class=_value(entry.device_class),
                    capabilities=capabilities,
                    # Availability gates what a future executor may consume;
                    # discovery itself never turns this into authority.
                    actionable_capabilities=capabilities if available else (),
                    supported_color_modes=classification.supported_color_modes,
                    status=status,
                    available=available,
                    explicit_controlled=entry.entity_id
                    in self.explicit_controlled_entity_ids,
                    discovered_candidate=entry.entity_id
                    not in self.explicit_controlled_entity_ids,
                ),
            )

        diagnostics = self._build_seed_diagnostics(area_ids)
        return tuple(entities), diagnostics

    def _build_seed_diagnostics(
        self,
        area_ids: set[str],
    ) -> tuple[DiscoveryDiagnostic, ...]:
        """Retain actionable diagnostics for missing or excluded configured seeds."""
        diagnostics: list[DiscoveryDiagnostic] = []
        for entity_id in self.seed_entity_ids:
            entry = self._entity_registry.async_get(entity_id)
            explicit = entity_id in self.explicit_controlled_entity_ids
            if entry is None:
                diagnostics.append(
                    DiscoveryDiagnostic(
                        entity_id=entity_id,
                        status=STATUS_MISSING,
                        reason="seed_not_registered",
                        area_id=None,
                        explicit_controlled=explicit,
                    ),
                )
                continue
            device = self._device_for(entry)
            area_id = effective_area_id(entry, device)
            if entry.disabled:
                diagnostics.append(
                    DiscoveryDiagnostic(
                        entity_id=entity_id,
                        status=STATUS_DISABLED,
                        reason="seed_disabled",
                        area_id=area_id,
                        explicit_controlled=explicit,
                    ),
                )
            elif entry.hidden:
                diagnostics.append(
                    DiscoveryDiagnostic(
                        entity_id=entity_id,
                        status=STATUS_HIDDEN,
                        reason="seed_hidden",
                        area_id=area_id,
                        explicit_controlled=explicit,
                    ),
                )
            elif area_id is None:
                diagnostics.append(
                    DiscoveryDiagnostic(
                        entity_id=entity_id,
                        status=STATUS_UNASSIGNED,
                        reason="seed_has_no_effective_area",
                        area_id=None,
                        explicit_controlled=explicit,
                    ),
                )
            elif area_id not in area_ids:
                diagnostics.append(
                    DiscoveryDiagnostic(
                        entity_id=entity_id,
                        status=STATUS_UNASSIGNED,
                        reason="seed_area_not_monitored",
                        area_id=area_id,
                        explicit_controlled=explicit,
                    ),
                )
        return tuple(diagnostics)


DiscoveryCoordinator = EntityDiscoveryCoordinator

__all__ = [
    "DiscoveryCoordinator",
    "DiscoveryDiagnostic",
    "EntityClassification",
    "EntityDiscoveryCoordinator",
    "EntityInventory",
    "InventoryChange",
    "InventoryMove",
    "InventoryRename",
    "InventorySnapshot",
    "classify_entity",
    "effective_area_id",
]
