---
title: Toothless intelligence rollout
icon: lucide/house-plug
---

# Toothless intelligence rollout

This is the operational plan for commissioning the continuously adapting
lighting behavior layer on Toothless. The general model and safety contract are
defined in [Context-aware intelligence architecture](intelligence-architecture.md).

## Intended behavior

The system learns attributable human `on` and `off` actions for each light-like
entity and room. It also learns explicit Good Night actions. It continuously
combines multiple time horizons with occupancy, presence, openings, arrival,
home/away, weekday/weekend/public-holiday state, media type and app, weather,
sun/daylight, illuminance, solar/energy, and alarm state.

A quick human reversal after an automatic action is strong negative feedback.
An unchanged automatic action becomes only weak positive evidence after its
observation window. Repeated corrections suppress that action/context while the
online model decays stale evidence and continues learning. Any accepted human
or physical light action also starts a persisted 30-minute per-entity hold, so
the model learns the choice without immediately fighting it.

## Capability and ownership rules

- Native `light` entities can learn power behavior whether they are dimmable or
  on/off-only. Brightness is used only when Home Assistant reports it.
- A `switch` is eligible only when conservative registry metadata identifies a
  primary light fixture. Maintenance controls, detached inputs, plugs,
  appliances, pumps, inverters, and other loads are excluded.
- If one device exposes both `light` and `switch`, the native `light` wins.
- Covers, garage doors, door/window sensors, alarm panels, media devices,
  weather, and solar entities are evidence only and are never behavior
  actuators.
- New, removed, renamed, unavailable, and area-moved entities are reconciled
  automatically. An area move resets room-specific model state instead of
  carrying a learned behavior into the new room.

## Commissioning sequence

1. Back up the live component, configuration, dashboard, and automations.
2. Install the reviewed fork and validate Home Assistant configuration.
3. Configure a South Africa holiday-only Workday sensor so public holidays use
   weekend behavior while retaining holiday provenance.
4. Start a persisted minimum seven-day `shadow_learning` session with
   auto-promotion enabled. During this phase the learned behavior executor must
   make zero power-state calls. The separately configured deterministic
   brightness baseline may adjust only an already-on dimmable light or enrich
   a caller-owned bare turn-on.
5. Verify the dashboard shows fresh context, discovered candidates, accepted and
   rejected observations, model support/confidence, proposals, corrections,
   suppressions, deadline, and promotion reason.
6. Only after observations are visibly being recorded, disable the competing
   ordinary lighting automations. Keep Good Night, alarm, safety, device-health,
   and unrelated non-light automations active so they remain valid semantic and
   safety context.
7. At the persisted deadline, promotion is automatic only if minimum samples,
   confidence, freshness, capability, availability, manual-hold, and safety
   gates pass. A failed gate leaves the house in shadow and exposes the reason.
8. After promotion, monitor unexpected actions, corrections, duplicate-command
   rate, unavailable-device behavior, and manual-hold violations. Any safety or
   repeated-correction regression returns the affected context to suppression
   or the deployment to shadow.

## Toothless commissioning baselines

Two narrow baselines remain active during the seven-day shadow period. They are
deliberately kept outside the learned actuator so that useful existing behavior
continues without fabricating human training samples.

### Living-room lamp and kitchen cabinet strip

The two dimmable ambient fixtures use the fork's native shadow-baseline path.
The tanh solar trajectory is the normal brightness envelope throughout the
day, including the evening descent. A room estimate derived from sun, cloud,
and trustworthy lux may lower that envelope, producing this bounded target:

`target = clamp(room estimate, 15%, current tanh target)`

The native controller enforces the following invariants:

- it never turns either fixture on or off;
- a bare Home Assistant turn-on receives the current bounded target in the same
  operation; a physical turn-on receives it immediately after the state change
  is observed. Both use a one-second initial hardware transition so the fixture
  does not linger at its previous level;
- an explicit scene or command that includes brightness remains authoritative;
- subsequent curve movement uses 90-second evaluations and 45-second hardware
  transitions, while the multi-hour tanh function—not those command
  transitions—defines the perceptual trajectory;
- a human brightness correction starts a 30-minute attribute-level hold before
  adaptation resumes;
- it uses `sensor.living_room_lamp_target_brightness` as the room estimate and
  enforces a 15–30% usable envelope;
- the darkness helper combines descending sun elevation, cloud cover, and the
  lowest trustworthy illuminance reading; the kitchen FP300 reading is ignored
  while the kitchen ceiling is on because that fixture contaminates the sensor;
- because power state never changes, human `on` and `off` actions remain clean
  observations for the learner.

The versioned
[`ambient_brightness_when_already_on.yaml`](../examples/toothless/ambient_brightness_when_already_on.yaml)
blueprint is retained as a rollback reference, but must remain disabled while
the native controller is active so two independent brightness loops cannot
compete.

### Garage light

The existing garage automations remain the operational prior while the model is
in shadow. Door movement or reliable occupancy turns the on/off-only garage
light on; occupancy clear starts a bounded hold; expiry may turn it off only
when the garage is closed, occupancy is clear, and no manual hold exists. A
separate invariant keeps the light on while the garage door is open and keeps
the detached wall-switch relay powered.

The whole-house discovery layer independently includes `light.garage_light`
and the garage opening/occupancy entities in its behavior context. It does not
treat the garage cover as an actuator. Attributable human changes and overrides
are learned continuously, including the day type and time horizons in effect;
automation-originated state changes do not masquerade as human preference. This
lets the learned policy eventually refine the operational prior without losing
the safe behavior that already works during commissioning.

## Release and rollback gates

Release requires the full test suite, static checks, configuration validation,
a clean restart, no new integration errors, correct live entity classification,
and a verified zero learned-power-call shadow interval. Expected deterministic
brightness calls must be attributable only to already-on lights or intercepted
external turn-ons. Rollback consists of disabling the native shadow baseline,
re-enabling the saved lighting automations, disabling intelligence/auto-
promotion, restoring the backed-up component and configuration, and restarting
Home Assistant. Local model storage can be retained for diagnosis or deleted
explicitly; it never contains credentials or raw event objects.
