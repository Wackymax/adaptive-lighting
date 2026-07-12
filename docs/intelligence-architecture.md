---
icon: lucide/brain-circuit
---

# Context-aware intelligence architecture

This document is the source of truth for the fork's intelligence boundary, safety
rules, and rollout plan. It describes how future context-aware behavior must fit
around the existing Adaptive Lighting component.

## Status: foundation implemented, activation staged

The repository currently contains a deterministic lighting foundation and a
Home-Assistant-independent intelligence foundation:

- sun-position-based brightness and color calculations;
- configured brightness and color limits;
- sleep mode and scheduled transitions;
- interception of relevant Home Assistant light calls;
- manual-control detection, manual-control reset services, and manual-control
  events;
- capability-aware handling of the brightness and color attributes exposed by a
  Home Assistant light;
- normalized context signals and snapshots with availability, freshness,
  confidence, and provenance;
- deterministic intent resolution with explicit priority and rejected
  alternatives;
- a bounded policy engine that produces a target and reasons without side
  effects;
- structured, presentation-neutral decision explanations; and
- transparent local preference and transition-prediction primitives suitable
  for replay and shadow evaluation.

Those behaviors are active when a user configures Adaptive Lighting. They are
not a machine-learning system and they do not claim to infer a person's
physiology, attention, or sleep state.

The context, intent, policy, explanation, and feedback boundaries below are the
implemented architectural contract for extending that foundation. The Home
Assistant integration has an opt-in intelligence configuration and read-only
preview/explanation seam; `intelligence_enabled` defaults to `false` and
`intelligence_shadow_mode` defaults to `true`. The pure learner and predictor
are not autonomous actuators, and live canary execution, durable HA-backed
learning, and general activation remain later rollout stages.

## Processing model

The control path is deliberately separated into five stages:

```text
context -> intent -> policy -> capability / executor -> explanation / feedback
```

Each stage has one job. A later stage may reject or narrow an earlier result,
but it must not silently broaden it.

| Stage | Responsibility | Foundation status |
| --- | --- | --- |
| **Context** | Normalize time, sun position, light state, service-call context, sensor readings, freshness, and quality. | Implemented as pure context signal/snapshot types. Home Assistant adapters still need to provide truthful source, age, and contamination details. |
| **Intent** | Express a conservative desired outcome such as “maintain configured ambient brightness”, “wind down”, “respect the manual hold”, or “do nothing”. | Implemented as deterministic intent resolution with explicit priorities, evidence, and rejected alternatives. |
| **Policy** | Apply priority, hard limits, manual-control invariants, confidence gates, and rollout mode to an intent. | Implemented as a side-effect-free bounded policy decision. The existing light executor remains the baseline authority. |
| **Capability / executor** | Check what the target light can actually accept, build the smallest valid Home Assistant command, and execute or decline it. | Baseline adaptation already distinguishes brightness and color attributes. Intelligence decision-to-command wiring and a reusable capability record remain gated work. |
| **Explanation / feedback** | Record why a decision was made, what was proposed, what happened, and whether a user or device overrode it. | Structured explanations and read-only preview/explain events exist. Durable feedback persistence and live outcome correlation are later work. |

Feedback may inform a later context record, but it must never bypass policy or
manual-control checks.

## Context contract

Context is an evidence record, not an instruction. Every input used by a future
policy should carry, where available:

- source and entity identity;
- value and unit;
- observation time and age;
- availability (`on`, `off`, `unknown`, or `unavailable` as applicable);
- quality and trust status;
- whether the value may have been caused by a previous light command; and
- the configuration or policy version that consumed it.

Useful inputs include sun position, time, current light state, configured
schedule, sleep mode, recent Home Assistant service calls, manual-control state,
occupancy or motion, and illuminance. None of these inputs should be treated as
equivalent: motion is not occupancy, a stale state is not a measurement, and a
lux reading is not a spectral measurement.

When context is incomplete, the safe result is an explicit `unknown` or a
no-op. A missing sensor must not be converted into “the room is empty” or
“the light should turn on”.

## Intent and policy separation

An intent says what outcome would be useful. A policy decides whether that
outcome is allowed and how strongly it may affect a device. For example:

| Context | Possible intent | Policy result |
| --- | --- | --- |
| Sun position changes while a configured light is on | Follow the configured daylight curve | Apply only supported attributes within configured limits. |
| A person changes brightness manually | Preserve the user's chosen brightness | Hold brightness; do not overwrite it while manual control is active. |
| Motion is detected but the light is off | Observe activity | No automatic turn-on from a prediction alone. An existing user automation remains the authority for turn-on behavior. |
| A lux sensor reports a value immediately after a light change | Re-evaluate ambient conditions | Mark the sample as potentially contaminated and defer learning from it. |
| A device is unavailable or reports no usable capability | Maintain safety and recoverability | Do not issue an unsupported command; retain the baseline behavior. |

## Safety priority

All policy decisions use this order, from highest to lowest priority:

1. explicit user commands and safety-relevant Home Assistant state;
2. device availability, valid capabilities, and hard integration limits;
3. active manual-control holds;
4. configured minimums, maximums, sleep settings, and transition limits;
5. deterministic time and sun-position behavior; and
6. contextual recommendations or predictions.

Lower-priority intelligence cannot override a higher-priority rule. In
particular, a predicted preference cannot turn a light on, exceed a configured
maximum, issue a color command to a brightness-only device, or clear a manual
hold.

## Manual-control invariants

The existing manual-control model is a safety boundary for any future
intelligence layer:

- A user-supplied brightness or color change is authoritative for the affected
  attribute while its manual-control flag is active.
- A light being off is not permission for an intelligence layer to turn it back
  on. Turn-on behavior belongs to an explicit Home Assistant action or a
  separately reviewed user automation.
- A manual hold is never cleared as a side effect of a prediction, explanation,
  feedback write, or sensor update.
- A hold may be released by the documented service, the documented off/on
  reset behavior, or an explicitly configured `autoreset_control_seconds`
  policy. An automatic reset is therefore an explicit user choice, not a
  hidden intelligence behavior.
- Attribute-level holds must remain attribute-level when the user changes only
  brightness or only color (`pause_changed` behavior). A full pause remains
  available when that is the configured policy.
- Group behavior must preserve the existing Home Assistant context and group
  rules. Do not infer a user's manual action from an unrelated or stale group
  member event.

## Capability and executor rules

Before producing a command, an executor must check the current target's
capability surface. At minimum this includes:

- whether the entity is a controllable `light` rather than a generic switch;
- whether brightness is supported and which range or percentage representation
  is accepted;
- whether color temperature, RGB, or another color mode is supported;
- whether the entity is available and its state is fresh enough to act on;
- whether the proposed change is redundant, out of bounds, or too frequent; and
- whether a transition or grouped call is safe for that device path.

The smallest valid command wins. A capability failure produces a reasoned
no-op and an explanation; it does not fall back to a guessed attribute.

### Toothless capability boundary

The current Toothless dimmers in scope for this fork are brightness-only. They
can accept a brightness change, but they do not provide a color-temperature or
RGB control surface that this architecture can rely on. A `switch` entity is
not a dimmer merely because it controls a light circuit.

Ordinary illuminance in lux is also not a melanopic measurement. A lux sensor
measures photopic illuminance and does not provide the spectrum needed to
calculate melanopic equivalent daylight illuminance or another melanopic target.
Therefore this fork must not claim to set or verify a melanopic target from an
ordinary lux reading, and it must not infer one from brightness percentage.

## Sensor contamination and data quality

Sensor readings can be changed by the very action the system is considering.
The context layer must treat these cases as contamination rather than as clean
feedback:

- A lux sensor that sees a controlled light should be marked contaminated for a
  settling window after that light changes.
- A light state observed immediately after a command should be tagged with its
  originating service context where possible; it is not independent proof that
  the policy was correct.
- `unknown`, `unavailable`, stale, unitless, or out-of-range readings must not
  train a predictor or trigger a new device action.
- Motion, occupancy, presence, and door sensors must retain their distinct
  meanings. A motion event may justify a hold or observation, but it is not
  proof of occupancy unless the configured policy says so.
- Scene, group, automation, and external-device changes must be separated from
  user intent whenever Home Assistant context allows it.
- Samples affected by a manual change, a device recovery, an unavailable
  coordinator, or a simultaneous automation should be excluded from learning
  or explicitly labelled as confounded.

When contamination cannot be ruled out, the system falls back to the existing
deterministic behavior and reports the uncertainty.

## Privacy and local learning

Any later learning must be local to the Home Assistant installation:

- no cloud model, telemetry endpoint, account, or external credential is
  required;
- raw presence history should not be retained when a coarse aggregate is
  sufficient;
- records should contain entity IDs and derived measurements only when needed,
  with configurable retention and a local deletion path;
- learning must be opt-in and independently disableable from lighting control;
- logs and explanations must not include secrets, access tokens, API keys, or
  private network details; and
- exporting a diagnostic report must require an explicit user action and must
  redact sensitive values.

The current foundation has no automatically persisted or externally hosted
learner and sends no intelligence data to an external service. The local
learner and predictor are bounded pure primitives for replay and shadow use;
they must be given an explicit storage adapter, retention policy, and migration
or deletion behavior before they are enabled for live learning.

## Prediction caps

Prediction is a constrained suggestion, not a new authority. The pure policy
and prediction primitives already enforce some of these bounds. Until a runtime
prediction gate is connected to the light executor, all of the following remain
activation requirements and must not be interpreted as permission for
autonomous actuation:

- **Scope cap:** target only explicitly configured lights and only attributes
  that the capability check confirms.
- **Magnitude cap:** keep every proposed brightness change inside configured
  minimum and maximum limits and inside a small per-update delta.
- **Frequency cap:** do not issue commands more often than the configured
  adaptation interval or the device can safely absorb.
- **Confidence cap:** low-confidence, stale, contaminated, or conflicting
  context produces a no-op or shadow-only proposal.
- **Action cap:** predictions cannot turn lights on or off, change a manual
  hold, claim a melanopic target, or bypass an explicit user command.
- **Horizon cap:** a prediction expires when its context is older than its
  bounded horizon; it must not be replayed after a restart as if it were fresh.
- **Fallback cap:** if any cap or capability check fails, retain the existing
  deterministic behavior rather than inventing a replacement.

## Rollout phases

Rollout is a progression of evidence, not a single feature flag:

1. **Contract and baseline:** keep the deterministic foundation unchanged,
   document reason codes and safety invariants, and establish baseline metrics.
2. **Offline replay:** evaluate recorded, locally retained events or synthetic
   fixtures without connecting the evaluator to live device services.
3. **Shadow mode:** evaluate the same live context and record the proposed
   intent, policy decision, caps, and expected command, but issue no light
   service call and change no Home Assistant light state. Shadow records must be
   local, bounded, and easy to delete.
4. **Single-fixture canary:** enable execution for one brightness-only light
   with color adaptation disabled, strict caps, manual-control protection, and a
   clear rollback path.
5. **Small cohort:** expand only to comparable lights after canary metrics meet
   the agreed safety gates. Keep unrelated switches, unstable devices, and
   lights with ambiguous capabilities out of the cohort.
6. **Measured activation:** make the feature user-selectable only after the
   evidence shows that it does not worsen unexpected turn-ons, manual-control
   violations, command failures, or device responsiveness.

Shadow mode is an evaluator state, not a synonym for “quietly execute”. Any
future configuration name or service must make this distinction explicit.

## Metrics and release gates

Metrics should be compared with the deterministic baseline and segmented by
device, room, capability, and rollout phase. Useful measures include:

| Area | Metric | Interpretation |
| --- | --- | --- |
| Safety | Unexpected light-on events per operating hour | Detects accidental turn-ons and false state interpretation. |
| Manual control | Manual overrides after a proposal; manual-hold violations | A proposal is not useful if users repeatedly have to fight it. Any violation is a release blocker. |
| Reliability | Command failure, timeout, unavailable-target, and duplicate-command rates | Shows whether the executor is adding load or issuing invalid calls. |
| Stability | Commands per light-hour, oscillations, redundant changes, and rate-limit hits | Detects chattering and network pressure. |
| Context quality | Stale, unknown, contaminated, or conflicting samples | Measures whether the system knows when not to act. |
| Usefulness | Accepted proposals, explicit feedback, and return-to-baseline behavior | Indicates value without treating passive non-intervention as approval. |
| Privacy | External intelligence requests and retained-record count/age | The intended external request count is zero; retention must stay within policy. |

Lux may be used to evaluate sensor quality or photopic ambient context. It must
not be used as evidence that a melanopic target was achieved. A release gate
must fail if the feature increases unexpected turn-ons, violates a manual hold,
acts on stale or contaminated context, or produces unsupported commands.

## Current configuration versus planned activation

The existing public configuration surface remains available: configured lights,
brightness and color adaptation switches, brightness and color limits, intervals
and transitions, sleep mode, manual-control settings, and the documented
services. The intelligence foundation additionally defines these opt-in,
inert-by-default settings:

- `intelligence_enabled` (default `false`): enable context-intelligence target
  evaluation; it does not by itself turn lights on;
- `intelligence_shadow_mode` (default `true`): keep intelligence decisions
  read-only and prevent intelligence-originated light service calls;
- context selectors for occupancy, presence, illuminance, home, security,
  sleep, media, energy constraints, manual hold, and semantic intent; and
- bounded intent caps for task, ambient, video, night, and prelight brightness.

The `adaptive_lighting.preview` and `adaptive_lighting.explain` service seams
publish read-only intelligence decisions/events. They are useful for diagnosis
and shadow evaluation; they are not a promise that a light will be changed.

The following remain planned activation controls rather than supported current
options: durable learning enablement, a prediction horizon, a per-update
prediction delta, a full capability registry, and a melanopic target. Do not
add names for those controls to a user's `configuration.yaml` until their
runtime implementation, schema, persistence, migration, and tests exist.

## Toothless example and safe starting point

The Toothless installation is a useful boundary test rather than a promise of
automatic behavior. A previous live check found Adaptive Lighting installed
but with no active Adaptive Lighting configuration, so the component was not
controlling any lights through an Adaptive Lighting switch.

The practical first cohort is limited to known brightness-capable light
entities such as `light.living_room_lamp` and
`light.kitchen_cabinet_strip`, after rechecking their live capabilities. The
brightness-only boundary means:

- enable or evaluate brightness adaptation only;
- keep color adaptation disabled unless a future device actually exposes and
  passes a color-capability check;
- do not treat `switch.dining_room_light` as a dimmable light; and
- use an illuminance entity such as `sensor.kitchen_motion_illuminance` only as
  photopic context, with contamination handling and no melanopic claim.

For a Toothless rollout, the safe order is to observe existing manual,
motion, daylight, and device-availability behavior; run a local shadow
evaluation; canary one stable brightness-only entity; and keep the existing
manual-control and explicit automation paths authoritative. No credential,
host address, token, or private network detail belongs in this documentation.

## Related documentation

- [Configuration reference](configuration.md)
- [Manual control](advanced/manual-control.md)
- [Services](services.md)
- [Fork ownership and upstream synchronization](fork-maintenance.md)
