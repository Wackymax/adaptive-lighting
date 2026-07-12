---
icon: lucide/brain-circuit
---

# Context-aware intelligence architecture

This document is the source of truth for the fork's intelligence boundary, safety
rules, implementation contract, and staged rollout. It describes how an
interpretable, continuously adapting behavior layer fits around the existing
Adaptive Lighting component.

## Status: foundation implemented, activation staged

The repository currently contains a deterministic lighting foundation, a
Home-Assistant-independent decision core, and a local commissioning adapter:

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
- transparent local preference and behavior-prediction primitives, including
  bounded temporal features, online updates, and exponential forgetting,
  suitable for replay and shadow evaluation;
- local, private seven-day shadow-learning state with a persisted deadline,
  sample counts, confidence, and promotion gates; and
- live entity discovery/reconciliation that keeps context inventory separate
  from the existing deterministic Adaptive Lighting light list.

The deterministic baseline remains authoritative until the intelligence layer
has enough evidence to propose a bounded change. The learning system is an
online preference and behavior model, not a black-box claim about a person's
physiology, attention, or sleep state. It learns only from attributable,
quality-checked local observations and can be reset or deleted.

The context, intent, policy, explanation, feedback, training, and discovery
boundaries below are the implemented architectural contract. The Home
Assistant integration has an opt-in intelligence configuration and read-only
preview/explanation seam; `intelligence_enabled` defaults to `false` and
`intelligence_shadow_mode` defaults to `true`. Training is separately opt-in
with `intelligence_training_enabled`; an enabled training session is forced to
zero-actuation shadow mode until its gates pass. Auto-promotion is separately
opt-in and must never be read as permission to bypass capability, freshness,
manual-control, or safety gates.

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

## Behavior model and feedback loop

The system learns behavior at several temporal horizons instead of treating a
single threshold as a household rule:

| Horizon | Examples | Use in a decision |
| --- | --- | --- |
| Cyclic time | Time-of-day sine/cosine features and coarse morning/day/evening/night buckets | Capture recurring daily behavior without treating midnight as far from 23:59. |
| Calendar | Weekday, weekend, and public holiday | Public holidays retain their provenance but intentionally use weekend behavior. |
| Event recency and dwell | Time since motion, opening, arrival, media, or a prior light action; how long a state persists | Separate a current event from stale residue and a durable preference. |
| Household state | Home/away and recent arrival | Permit a low, bounded welcome or handoff context only when the evidence is fresh. |
| Environment | Weather, daylight/sun position, illuminance, and a solar/PV proxy | Estimate ambient conditions and confidence; ordinary lux remains photopic context, not melanopic exposure. |
| Activity | Media category and application, semantic intent, openings, motion, and presence | Distinguish video, music, task, ambient, and movement through a space. |

Room-local evidence is kept separate from house-wide evidence. Motion,
occupancy, presence, opening recency, and dwell from one area cannot satisfy an
actuation gate for a light in another area. Home/away, arrival, calendar,
weather, solar, alarm, and media may remain house-wide context, while each
per-light model receives only the matching area overlay. If an entity moves to
another area, its room-specific model is reset and relearned.

The online loop is deliberately asymmetric:

1. A high-confidence, well-supported, fresh context can produce a bounded
   proposed action. The proposal expires; it is never a permanent command.
2. A quick manual reversal after an automatic proposal is strong negative
   feedback. It carries more learning weight than passive non-intervention.
3. An action that remains accepted for its observation window is weak positive
   feedback. It increases support slowly and cannot make a sparse context look
   certain.
4. Repeated corrections in the same context enter suppression/cooldown. The
   system backs off and returns to the deterministic baseline until new,
   durable evidence justifies reconsideration.

Every accepted human or physical on/off action also starts a persisted
30-minute per-entity hold. The action is learned immediately, but no behavior
proposal may fight it during that interval. This covers native lights and
switch-backed fixtures equally, survives restart and registry reconciliation,
and is separate from Good Night automation or autonomous feedback.

Confidence is not a single model score. A proposal must pass support,
confidence, freshness, expiry, capability, availability, and safety gates. A
stale or contaminated context is a no-op/shadow-only result even when the
historical model is confident.

### Semantic routines and safety context

Good Night is a learnable semantic routine. An explicit `good_night` routine can
teach the coordinated off/path behavior of multiple light-like targets, while
remaining bounded by the normal policy and capability rules. It is not merely
an arbitrary automation label.

Alarm and other safety-triggered actions are different: they are safety
context, not household preference. Alarm-triggered actions, emergency paths,
grid emergencies, and similar safety events are excluded as training targets
and must retain their explicit safety authority. A safety event may override
ordinary media, arrival, or ambient context, but it must not contaminate the
preference model.

### Continuous adaptation after commissioning

The seven-day shadow/learning phase is a minimum evidence period, not a
one-time batch-training job. After the deadline, auto-promotion is considered
only when minimum samples, confidence, durability, data quality, and safety
gates pass. In the active phase the model continues to update online from
bounded evidence, applies forgetting/decay, and re-enters a conservative
shadow or suppressed state when drift or repeated corrections reduce support.
Continuous adaptation means gradual recalibration under constraints; it does
not mean an unconstrained model can rewrite device ownership or safety logic.

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

Useful inputs include cyclic time, weekday/weekend/public-holiday calendar
state, current light state, configured schedule, sleep mode, recent service
calls, manual-control state, occupancy, motion, presence, openings, arrival,
home/away, dwell and event recency, illuminance, weather, sun/daylight, solar
or PV proxy, media type and app, and semantic routines. None of these inputs
should be treated as equivalent: motion is not occupancy, a stale state is not
a measurement, and a lux reading is not a spectral measurement.

Doors, windows, and garage covers are context-only. Discovery may report their
state, recency, and area, but the intelligence layer never actuates a cover,
door, garage, lock, valve, or window. They can explain an arrival, opening, or
transition; they cannot become a predicted actuator because they happen to be
near a light.

When context is incomplete, the safe result is an explicit `unknown` or a
no-op. A missing sensor must not be converted into “the room is empty” or
“the light should turn on”.

### Discovery and reconciliation

The discovery coordinator observes the Home Assistant entity/device registry
and reconciles its bounded inventory as entities are added, removed, renamed,
or moved between areas. Each refresh publishes the revision, reason, current
capabilities, availability, and deltas. A newly discovered native light, or a
conservatively classified switch-backed fixture, can become a behavior-model
candidate without being added to the deterministic brightness/color adaptation
list. Every candidate still needs a supported light capability, a stable area
or explicit configuration, availability, fresh evidence, and the active
execution gates. A renamed or
moved entity is reconciled immediately; an area move resets its room-specific
model, while a capability or availability change forces a no-op until
revalidated.

Discovery does not grant immediate permission. It may create a shadow behavior
candidate only for a native light or conservatively classified primary
light-switch load; that candidate still has to earn every commissioning and
execution gate. Covers, doors, and garages remain context-only in every delta
and are never promoted to actuators.

## Intent and policy separation

An intent says what outcome would be useful. A policy decides whether that
outcome is allowed and how strongly it may affect a device. For example:

| Context | Possible intent | Policy result |
| --- | --- | --- |
| Sun position changes while a configured light is on | Follow the configured daylight curve | Apply only supported attributes within configured limits. |
| A person changes brightness manually | Preserve the user's chosen brightness | Hold brightness; do not overwrite it while manual control is active. |
| Motion is detected but the light is off | Observe activity | Motion alone grants no authority. After commissioning, only a fresh, well-supported per-light prediction with home/presence and all execution gates may propose power-on. |
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
particular, a prediction score alone cannot turn a light on, exceed a configured
maximum, issue a color command to a brightness-only device, actuate a non-light
domain, or clear a manual hold.

## Manual-control invariants

The existing manual-control model is a safety boundary for any future
intelligence layer:

- A user-supplied brightness or color change is authoritative for the affected
  attribute while its manual-control flag is active.
- A light being off is not by itself permission to turn it back on. An active
  behavior proposal also requires attributable training, fresh home and
  occupancy/recent-arrival context, per-light support and confidence, no
  pending/corrected proposal, and every safety/capability gate.
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

Brightness is a conditional capability, not a universal property of a light.
The executor supports both:

- **dimmable lights**, where brightness is changed only when Home Assistant
  reports a brightness-capable mode or live brightness attribute; and
- **on/off-only lights**, where the valid action is power state only and no
  brightness percentage is invented.

An on/off-only `switch` is not automatically a light target. Its registry and
live metadata must conservatively identify the primary load as a light fixture,
without appliance or maintenance-control markers, before it can participate in
any light policy. Conversely, a brightness-capable light still cannot be used
for color-temperature or RGB adaptation unless those capabilities are reported.

### Toothless capability boundary

The current Toothless dimmers in scope for this fork are brightness-only. They
can accept a brightness change, but they do not provide a color-temperature or
RGB control surface that this architecture can rely on. Toothless also has
switch-backed on/off-only fixtures. A `switch` entity is not a dimmer merely
because it controls a light circuit.

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

## Privacy, bounded storage, and drift

Learning is local to the Home Assistant installation:

- no cloud model, telemetry endpoint, account, or external credential is
  required;
- raw presence history should not be retained when a coarse aggregate is
  sufficient;
- records should contain entity IDs and derived measurements only when needed,
  with bounded retention, a local deletion path, and versioned storage;
- learning must be opt-in and independently disableable from lighting control;
- logs and explanations must not include secrets, access tokens, API keys, or
  private network details; and
- exporting a diagnostic report must require an explicit user action and must
  redact sensitive values.

The training adapter stores only compact, JSON-safe local state: phase and
deadline, sample/rejection counts, day-type counts, bounded learner state,
pending durability candidates, and the last diagnostic sample. It uses
Home Assistant's private local storage and sends no intelligence data to an
external service. Raw event objects are not persisted.

The online model is intentionally interpretable: bounded local preference
offsets plus a per-entity online logistic action model with explicit support,
freshness, confidence factors, expiry, feature contributions, and provenance,
rather than an opaque neural policy. Bounded updates, minimum support,
confidence thresholds, context-age limits, durability windows, human-action
holds, rapid-toggle rejection, and correction cooldowns keep sparse data from
becoming aggressive behavior. Continuous adaptation uses exponential
decay/forgetting and re-evaluation of recent evidence so stale preferences do
not become permanent policy. A reset is a supported safety operation, not a
migration workaround.

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
- **Action cap:** only the dedicated behavior executor may turn a classified
  light-like entity on or off, and only in active phase. Predictions can never
  change a manual hold, actuate covers/doors/locks/valves, claim a melanopic
  target, or bypass an explicit user command.
- **Horizon cap:** a prediction expires when its context is older than its
  bounded horizon; it must not be replayed after a restart as if it were fresh.
- **Fallback cap:** if any cap or capability check fails, retain the existing
  deterministic behavior rather than inventing a replacement.

## Staged rollout

Rollout is a progression of evidence, not a single feature flag:

1. **Contract and baseline:** keep deterministic sun/manual/safety behavior
   authoritative, establish per-light and per-intent baseline metrics, and
   verify that the entity inventory distinguishes actuators from context.
2. **Offline replay:** evaluate local or synthetic records without connecting
   the evaluator to live device services. Confirm explanations, rejected
   alternatives, capability handling, learnable Good Night propagation, and
   alarm/safety exclusion.
3. **Minimum seven-day shadow/learning:** enable
   `intelligence_training_enabled` with a training duration of at least seven
   days. The training session records bounded local evidence and proposals,
   but zero Adaptive Lighting light service calls are permitted. The shadow
   actuation block remains in force during the entire phase, including manual
   refreshes, restarts, and deadline processing.
4. **Gate evaluation and auto-promotion:** at or after the persisted deadline,
   evaluate minimum accepted samples, confidence, durable observations,
   freshness/contamination quality, no manual-hold violations, no unsupported
   commands, and stable explanations. Only when all gates pass may
   `intelligence_auto_promote` move the session to active. If a gate fails,
   remain in a conservative shadow/blocked state and expose the reason.
5. **Continuous active adaptation:** after promotion, continue updating the
   local model online, but keep every proposal behind support, confidence,
   freshness, capability, magnitude, expiry, and manual-control gates. Quick
   reversals carry strong negative weight; accepted actions carry weak positive
   weight; repeated corrections trigger suppression/cooldown and fall back to
   the deterministic baseline. Drift or reduced freshness can send a context
   back to shadow evaluation.
6. **Per-fixture earned activation:** whole-house candidates remain independent
   models. A fixture with sparse, stale, corrected, unavailable, or ambiguous
   evidence stays shadow-only even when another fixture qualifies. Keep covers,
   doors, garages, appliance switches, integration-level light groups, unstable
   devices, and ambiguous capabilities out of the actuator cohort.
7. **Measured activation:** make the feature user-selectable only after
   evidence shows no increase in unexpected turn-ons, manual-control
   violations, command failures, repeated corrections, or device
   responsiveness problems.

Shadow mode is an evaluator state, not a synonym for “quietly execute”. A
commissioned deployment keeps `intelligence_shadow_mode: true`;
auto-promotion is the only documented path out of that block, and only after
the gates above.

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
| Learning quality | Accepted human samples, durable corrections, quick reversals, weak acceptances, superseded samples, and suppression/cooldown time | Shows whether the loop is learning the right amount and backing off when corrected. |
| Usefulness | Accepted proposals, explicit feedback, and return-to-baseline behavior | Indicates value without treating passive non-intervention as approval. |
| Discovery | Added, removed, renamed, moved, unavailable, and context-only entities | Detects inventory drift while keeping non-light context entities non-actuating. |
| Privacy | External intelligence requests and retained-record count/age | The intended external request count is zero; retention must stay within policy. |

Lux may be used to evaluate sensor quality or photopic ambient context. It must
not be used as evidence that a melanopic target was achieved. A release gate
must fail if the feature increases unexpected turn-ons, violates a manual hold,
acts on stale or contaminated context, or produces unsupported commands.

## Current configuration and staged activation

The existing public configuration surface remains available: configured lights,
brightness and color adaptation switches, brightness and color limits, intervals
and transitions, sleep mode, manual-control settings, and the documented
services. The intelligence foundation additionally defines these opt-in,
inert-by-default settings:

- `intelligence_enabled` (default `false`): enable context-intelligence target
  evaluation; it does not by itself turn lights on;
- `intelligence_shadow_mode` (default `true`): keep intelligence decisions
  read-only and prevent intelligence-originated light service calls;
- `intelligence_training_enabled` (default `false`): enable local shadow
  learning and its private bounded store;
- `intelligence_training_days` (default `7`): minimum operational rollout is
  seven days, even though the schema permits other values for tests or
  controlled experiments;
- `intelligence_auto_promote` (default `false`): permit active phase only after
  the persisted deadline and all promotion gates pass;
- `intelligence_minimum_samples` (default `8`) and
  `intelligence_minimum_confidence` (default `0.8`): promotion gates; and
- `intelligence_durability_seconds` (default `120`): how long a manual
  brightness correction must persist before it becomes learning evidence;
- context selectors for occupancy, presence, illuminance, home, security,
  sleep, media, energy constraints, manual hold, and semantic intent; and
- bounded intent caps for task, ambient, video, night, and prelight brightness.

The `adaptive_lighting.preview` and `adaptive_lighting.explain` service seams
publish read-only intelligence decisions/events. They are useful for diagnosis
and shadow evaluation; they are not a promise that a light will be changed.

The behavior learner's richer proposal/outcome bookkeeping remains a bounded
local model contract; it is not a replacement for the existing explicit
Home Assistant safety automations. Do not add new context-selector names or
assume a sensor exists merely because the model can represent that feature.
When a signal is absent, discovery reports it as missing/unavailable and the
policy falls back rather than guessing.

## Toothless example and safe starting point

The Toothless installation is a useful boundary test rather than a promise of
automatic behavior. A previous live check found Adaptive Lighting installed
but with no active Adaptive Lighting configuration, so the component was not
controlling any lights through an Adaptive Lighting switch.

The practical first cohort is limited to known brightness-capable light
entities such as `light.living_room_lamp` and
`light.kitchen_cabinet_strip`, after rechecking their live capabilities. The
known `light.kitchen_ceiling_light` is also brightness-capable in Home
Assistant. The first two entities are the deterministic Adaptive Lighting
brightness cohort; the behavior layer observes every safely classified
whole-house candidate during shadow, including on/off-only fixtures, without
issuing power commands. The brightness-only boundary means:

- enable or evaluate brightness adaptation only;
- keep color adaptation disabled unless a future device actually exposes and
  passes a color-capability check;
- treat `switch.dining_room_light` as an on/off-only fixture, never as a
  dimmable light; and
- use an illuminance entity such as `sensor.kitchen_motion_illuminance` only as
  photopic context, with contamination handling and no melanopic claim.

For a Toothless rollout, the safe order is to observe existing manual, motion,
daylight, and device-availability behavior; run the full seven-day local shadow
phase; disable only competing ordinary lighting automations after observation
is verified; and keep manual control, Good Night, alarm, safety, and
device-health paths authoritative. No credential, host address, token, or
private network detail belongs in this documentation.

## Research rationale and evidence boundary

The design favors small, interpretable, temporally aware local models for
three reasons:

- Fatima et al. describe sequential behavior extraction and future-action
  prediction from recognized smart-home activity. That supports retaining
  event order, recency, and bounded expiry rather than treating every sensor
  state as an independent rule: [A Unified Framework for Activity
  Recognition-Based Behavior Analysis and Action Prediction in Smart Homes
  (Fatima et al., 2013)](https://doi.org/10.3390/s130202682).
- Bouchabou et al. survey the irregular, heterogeneous sensor and temporal
  challenges of smart-home human-activity recognition. That supports explicit
  provenance, freshness, multiple temporal horizons, conservative missing-data
  handling, and local privacy boundaries: [A Survey of Human Activity
  Recognition in Smart Homes Based on IoT Sensors Algorithms (Bouchabou et
  al., 2021)](https://doi.org/10.3390/s21186037).
- Das et al. evaluate explanations for smart-home activity recognition and
  show why a user-facing system needs to expose the evidence behind an
  automated result. That supports decision reasons, confidence, rejected
  alternatives, and quick manual reversal as visible feedback: [Explainable
  Activity Recognition for Smart Home Systems (Das et al., 2023)](https://doi.org/10.1145/3561533).

These references are primary-paper DOI/metadata and abstract-level rationale
for this rollout. This document does not claim a full-text review of any paper
unless the full text was directly available and read; the references are not
evidence that this implementation reproduces their datasets, models, or
reported results.

## Related documentation

- [Configuration reference](configuration.md)
- [Manual control](advanced/manual-control.md)
- [Services](services.md)
- [Fork ownership and upstream synchronization](fork-maintenance.md)
