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
   auto-promotion enabled. During this phase the behavior executor must make
   zero light or switch service calls.
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

## Release and rollback gates

Release requires the full test suite, static checks, configuration validation,
a clean restart, no new integration errors, correct live entity classification,
and a verified zero-call shadow interval. Rollback consists of re-enabling the
saved lighting automations, disabling intelligence/auto-promotion, restoring the
backed-up component and configuration, and restarting Home Assistant. Local
model storage can be retained for diagnosis or deleted explicitly; it never
contains credentials or raw event objects.
