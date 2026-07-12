"""Tests for Adaptive Lighting entity discovery."""

import json  # noqa: I001

import pytest
from homeassistant.components.light import (
    ATTR_SUPPORTED_COLOR_MODES,
    ColorMode,
)
from homeassistant.const import STATE_UNAVAILABLE
from homeassistant.helpers import (
    area_registry as ar,
    device_registry as dr,
    entity_registry as er,
)

from homeassistant.components.adaptive_lighting.discovery import (
    EntityDiscoveryCoordinator,
    classify_entity,
    effective_area_id,
)


@pytest.fixture
def discovery_config_entry(hass):
    """Provide the config entry required by the vendored device registry."""
    from tests.common import MockConfigEntry  # noqa: PLC0415

    entry = MockConfigEntry(domain="adaptive_lighting_test")
    entry.add_to_hass(hass)
    return entry


def _device(device_registry, name: str, area_id: str, config_entry_id: str):
    device = device_registry.async_get_or_create(
        config_entry_id=config_entry_id,
        identifiers={("adaptive_lighting_test", name)},
        name=name,
    )
    return device_registry.async_update_device(device.id, area_id=area_id)


def _entity(
    entity_registry,
    domain: str,
    unique_id: str,
    *,
    area_id: str | None = None,
    device_id: str | None = None,
    device_class: str | None = None,
    capabilities: dict | None = None,
    disabled_by=None,
    hidden_by=None,
):
    entry = entity_registry.async_get_or_create(
        domain,
        "adaptive_lighting_test",
        unique_id,
        device_id=device_id,
        capabilities=capabilities,
        disabled_by=disabled_by,
        hidden_by=hidden_by,
    )
    return entity_registry.async_update_entity(
        entry.entity_id,
        area_id=area_id,
        device_class=device_class,
    )


async def test_classification_uses_vendored_ha_capabilities(
    hass,
    area_registry,
    device_registry,
    entity_registry,
    discovery_config_entry,
):
    """Classify lights and context entities from registry/state metadata."""
    area = area_registry.async_create("Living Room")
    device = _device(
        device_registry,
        "living",
        area.id,
        discovery_config_entry.entry_id,
    )
    light = _entity(
        entity_registry,
        "light",
        "dimmable",
        device_id=device.id,
        capabilities={ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS]},
    )
    motion = _entity(
        entity_registry,
        "binary_sensor",
        "motion",
        area_id=area.id,
        device_class="motion",
    )
    occupancy = _entity(
        entity_registry,
        "binary_sensor",
        "occupancy",
        area_id=area.id,
        device_class="occupancy",
    )
    opening = _entity(
        entity_registry,
        "binary_sensor",
        "opening",
        area_id=area.id,
        device_class="opening",
    )
    garage = _entity(
        entity_registry,
        "cover",
        "garage",
        area_id=area.id,
        device_class="garage",
    )
    illuminance = _entity(
        entity_registry,
        "sensor",
        "lux",
        area_id=area.id,
        device_class="illuminance",
    )
    media = _entity(
        entity_registry,
        "media_player",
        "tv",
        area_id=area.id,
    )
    holiday = _entity(
        entity_registry,
        "calendar",
        "south_africa_public_holidays",
    )

    hass.states.async_set(
        light.entity_id,
        "on",
        {ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS]},
    )
    for entry in (motion, occupancy, opening):
        hass.states.async_set(entry.entity_id, "off")
    hass.states.async_set(garage.entity_id, "closed")
    hass.states.async_set(illuminance.entity_id, "120")
    hass.states.async_set(media.entity_id, "idle")
    hass.states.async_set(
        holiday.entity_id,
        "off",
        {"friendly_name": "South Africa Public Holidays"},
    )

    assert classify_entity(light, hass.states.get(light.entity_id)).capabilities == (
        "light",
        "on_off_light",
        "dimmable_light",
    )
    assert classify_entity(light).dimmable
    assert (
        "motion"
        in classify_entity(motion, hass.states.get(motion.entity_id)).capabilities
    )
    assert (
        "occupancy"
        in classify_entity(occupancy, hass.states.get(occupancy.entity_id)).capabilities
    )
    assert (
        "opening"
        in classify_entity(opening, hass.states.get(opening.entity_id)).capabilities
    )
    assert {"cover", "opening", "garage_door"}.issubset(
        classify_entity(garage, hass.states.get(garage.entity_id)).capabilities,
    )
    assert {"illuminance", "context"}.issubset(
        classify_entity(
            illuminance,
            hass.states.get(illuminance.entity_id),
        ).capabilities,
    )
    assert {"media", "context"}.issubset(
        classify_entity(media, hass.states.get(media.entity_id)).capabilities,
    )
    assert {"holiday_calendar", "context"}.issubset(
        classify_entity(holiday, hass.states.get(holiday.entity_id)).capabilities,
    )

    coordinator = EntityDiscoveryCoordinator(
        hass,
        seed_entity_ids=[light.entity_id],
        explicit_controlled_entity_ids=[light.entity_id],
    )
    snapshot = await coordinator.start()

    assert snapshot.monitored_area_ids == (area.id,)
    assert snapshot.entity_ids == tuple(
        sorted(
            entry.entity_id
            for entry in (
                light,
                motion,
                occupancy,
                opening,
                garage,
                illuminance,
                media,
                holiday,
            )
        ),
    )
    discovered_light = next(
        item for item in snapshot.entities if item.entity_id == light.entity_id
    )
    assert discovered_light.explicit_controlled
    assert not discovered_light.discovered_candidate
    assert discovered_light.actionable_capabilities
    assert json.loads(json.dumps(snapshot.to_dict()))["revision"] == 1
    await coordinator.stop()


async def test_add_remove_move_and_disabled_exclusion(
    hass,
    area_registry,
    device_registry,
    entity_registry,
    discovery_config_entry,
):
    """Reconcile registry mutations and preserve entity-over-device area semantics."""
    living = area_registry.async_create("Living")
    bedroom = area_registry.async_create("Bedroom")
    seed = _entity(entity_registry, "light", "seed", area_id=living.id)
    bedroom_seed = _entity(
        entity_registry,
        "light",
        "bedroom_seed",
        area_id=bedroom.id,
    )
    candidate = _entity(entity_registry, "sensor", "candidate", area_id=living.id)
    disabled = _entity(
        entity_registry,
        "binary_sensor",
        "disabled",
        area_id=living.id,
        device_class="motion",
        disabled_by=er.RegistryEntryDisabler.USER,
    )
    hass.states.async_set(seed.entity_id, "off")
    hass.states.async_set(bedroom_seed.entity_id, "off")
    hass.states.async_set(candidate.entity_id, "1")
    hass.states.async_set(disabled.entity_id, "off")

    changes = []
    coordinator = EntityDiscoveryCoordinator(
        hass,
        seed_entity_ids=[seed.entity_id, bedroom_seed.entity_id],
        debounce_seconds=0,
        on_change=changes.append,
    )
    initial = await coordinator.start()
    assert candidate.entity_id in initial.entity_ids
    assert disabled.entity_id not in initial.entity_ids

    added = _entity(
        entity_registry,
        "binary_sensor",
        "added",
        area_id=living.id,
        device_class="presence",
    )
    hass.states.async_set(added.entity_id, "on")
    await hass.async_block_till_done()
    after_add = coordinator.snapshot
    assert [item.entity_id for item in after_add.added] == [added.entity_id]

    entity_registry.async_remove(added.entity_id)
    await hass.async_block_till_done()
    after_remove = coordinator.snapshot
    assert [item.entity_id for item in after_remove.removed] == [added.entity_id]
    assert added.entity_id not in after_remove.entity_ids

    entity_registry.async_update_entity(candidate.entity_id, area_id=bedroom.id)
    await hass.async_block_till_done()
    after_move = coordinator.snapshot
    assert len(after_move.moved) == 1
    assert after_move.moved[0].entity_id == candidate.entity_id
    assert after_move.moved[0].old_area_id == living.id
    assert after_move.moved[0].new_area_id == bedroom.id
    assert candidate.entity_id in after_move.entity_ids

    follow_device = _device(
        device_registry,
        "follows_device_area",
        living.id,
        discovery_config_entry.entry_id,
    )
    follows_device = _entity(
        entity_registry,
        "sensor",
        "follows_device_area",
        device_id=follow_device.id,
    )
    hass.states.async_set(follows_device.entity_id, "3")
    await hass.async_block_till_done()
    assert follows_device.entity_id in coordinator.snapshot.entity_ids
    device_registry.async_update_device(follow_device.id, area_id=bedroom.id)
    await hass.async_block_till_done()
    assert coordinator.snapshot.moved[0].entity_id == follows_device.entity_id
    assert coordinator.snapshot.moved[0].old_area_id == living.id
    assert coordinator.snapshot.moved[0].new_area_id == bedroom.id

    # The seed is still in Living, so a device-area change cannot broaden the
    # monitored cohort.  A direct entity override remains authoritative.
    device = _device(
        device_registry,
        "override",
        living.id,
        discovery_config_entry.entry_id,
    )
    override = _entity(
        entity_registry,
        "sensor",
        "override",
        device_id=device.id,
        area_id=living.id,
    )
    hass.states.async_set(override.entity_id, "2")
    assert (
        effective_area_id(entity_registry.async_get(override.entity_id), device)
        == living.id
    )
    device_registry.async_update_device(device.id, area_id=bedroom.id)
    await hass.async_block_till_done()
    override_item = next(
        item
        for item in coordinator.snapshot.entities
        if item.entity_id == override.entity_id
    )
    assert override_item.area_id == living.id
    area_registry.async_update(living.id, name="Living Renamed")
    await hass.async_block_till_done()
    assert any("area_name" in change.fields for change in coordinator.snapshot.changed)
    assert len(changes) >= 3
    await coordinator.stop()


async def test_unavailable_and_missing_entities_are_diagnostic_not_actionable(
    hass,
    area_registry,
    entity_registry,
):
    """Keep status while withholding unavailable capability from a future executor."""
    area = area_registry.async_create("Office")
    seed = _entity(entity_registry, "light", "seed", area_id=area.id)
    missing = _entity(entity_registry, "sensor", "missing", area_id=area.id)
    unavailable = _entity(
        entity_registry,
        "binary_sensor",
        "unavailable",
        area_id=area.id,
        device_class="motion",
    )
    hass.states.async_set(seed.entity_id, "off")
    hass.states.async_set(unavailable.entity_id, STATE_UNAVAILABLE)

    coordinator = EntityDiscoveryCoordinator(hass, seed_entity_ids=[seed.entity_id])
    snapshot = await coordinator.start()
    missing_item = next(
        item for item in snapshot.entities if item.entity_id == missing.entity_id
    )
    unavailable_item = next(
        item for item in snapshot.entities if item.entity_id == unavailable.entity_id
    )
    assert missing_item.status == "missing"
    assert not missing_item.available
    assert not missing_item.actionable_capabilities
    assert unavailable_item.status == "unavailable"
    assert not unavailable_item.available
    assert not unavailable_item.actionable_capabilities
    await coordinator.stop()


async def test_rename_and_stop_remove_all_runtime_work(
    hass,
    area_registry,
    entity_registry,
):
    """Publish rename deltas and do not refresh after lifecycle shutdown."""
    area = area_registry.async_create("Kitchen")
    seed = _entity(entity_registry, "light", "seed", area_id=area.id)
    rename_candidate = _entity(
        entity_registry,
        "sensor",
        "rename_candidate",
        area_id=area.id,
    )
    hass.states.async_set(seed.entity_id, "off")
    hass.states.async_set(rename_candidate.entity_id, "1")
    coordinator = EntityDiscoveryCoordinator(
        hass,
        seed_entity_ids=[seed.entity_id],
        debounce_seconds=0.01,
    )
    await coordinator.start()
    old_id = rename_candidate.entity_id
    entity_registry.async_update_entity(
        old_id,
        new_entity_id="sensor.renamed_candidate",
    )
    hass.states.async_set("sensor.renamed_candidate", "1")
    await hass.async_block_till_done()
    assert len(coordinator.snapshot.renamed) == 1
    assert coordinator.snapshot.renamed[0].old_entity_id == old_id
    assert coordinator.snapshot.renamed[0].new_entity_id == "sensor.renamed_candidate"

    revision = coordinator.snapshot.revision
    listeners_before = hass.bus.async_listeners()
    await coordinator.stop()
    listeners_after = hass.bus.async_listeners()
    for event_type in (
        er.EVENT_ENTITY_REGISTRY_UPDATED,
        dr.EVENT_DEVICE_REGISTRY_UPDATED,
        ar.EVENT_AREA_REGISTRY_UPDATED,
    ):
        assert (
            listeners_after.get(event_type, 0)
            == listeners_before.get(event_type, 0) - 1
        )
    entity_registry.async_get_or_create(
        "sensor",
        "adaptive_lighting_test",
        "after_stop",
    )
    await hass.async_block_till_done()
    assert coordinator.snapshot.revision == revision
    assert coordinator._refresh_task is None
    assert not coordinator.started
