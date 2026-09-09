# State, recommendation model, and evening investigation — 8 September 2026

## Implemented locally

- Recommendations use `gpt-5.6-luna`. The existing local OpenAI key can access
  that model (authenticated model lookup returned HTTP 200). The request retains
  web research, medium reasoning, structured output, and the existing spending
  controls. Model availability was checked; a live recommendation quality
  evaluation was not performed.
- The shared venue cache now includes the model and all research inputs: exact
  coordinates, area label, and query. Its new namespace ignores old Terra cache
  entries without deleting them. Identical requests still reuse research.
- Venue API responses use `no-store`; deliberate reuse is handled by the
  application cache. The browser's venue cache also includes the complete area.
- The browser's map cache includes participant names and exact coordinates,
  preventing stale map labels after a participant is renamed.

## What was reproduced

The Worker used to identify venue requests by coordinates rounded to three
decimal places and a lowercased query. It then returned the complete earlier
response, including that visitor's exact coordinates, area label, and wording.
A test using two visitor IPs and two nearby locations failed before the fix:
visitor B received visitor A's area. It now passes. A separate test confirms
renaming a participant updates map labels while retaining the model values.

These are real cache defects, but they do **not** establish the cause of the
reported participant data appearing across devices:

- All 3,538 cells in the currently shipped atlas have distinct coordinate pairs
  even when rounded to three decimal places. The current interface chooses
  venue areas from those cells, so the reproduced nearby-coordinate collision
  is not expected during ordinary use of this atlas.
- Venue results update venue cards and map markers. They do not write the
  participant array or its saved locations.
- The map response cache exists only inside the browser's JavaScript process.
  Its stale-name defect can affect the same browser session, not another device.

## Participant-state investigation remains open

`AppStateContext.tsx` reads and writes `equidistant:workspace:v2` in the browser's
local storage. Participant mutations come from local input, adding/removing
people, map placement, or reset. The production product view calculates the
travel surface locally. There is no participant-list synchronization endpoint,
WebSocket, storage-event bridge, or service worker in this source.

The live D1 schema contains analytics, authentication throttling, venue cache,
venue usage, and TravelTime usage tables. It contains no workspace table.
Protected HTML is marked `no-store`; the asset build contains static files,
not request-specific participant state. The Python development API's group
cache uses the complete request, including names, as its key and copies the
destination data before modification.

The provider test preserves profile A's participant changes, opens separate
empty storage, verifies that profile B does not receive those changes, then
restores A's storage and verifies its own saved state. This is a component-level
storage isolation test, not a reproduction across two physical devices.

The live site still presents its password gate in a fresh inspection browser;
a direct HTTP check also returned `Cache-Control: no-store`.
No live participant state was changed during this investigation. An unexpected
name/location, approximate incident date, and the actual site URL are still
needed to connect the reported observation to a particular release or request.
The incident is not considered fixed merely because the two cache defects are
fixed.

### Follow-up with specific participant examples

The user supplied two specific unexpected participant names and a Walthamstow
location. Neither name appears in the current source or in any of the exact
source commits attached to all 21 saved Sites versions (9–19 July 2026).
This rules out those names being the source-defined sample participants. It
does not prove that an archived deployment bundle matched its source exactly.
Walthamstow does appear in the station catalogue; a station label alone does not
establish where the associated participant name came from.

The shared-link artwork contains no participant names. A bounded sample of 30
recent Worker events had no matches, but recent logs cannot reconstruct an
undated earlier incident. The user subsequently confirmed actual Safari on
their phone, shortly after sharing the site with the other person. They
recognised one person's real location and learned a previously unknown location
from the data shown. Treat this as a reported privacy incident; the preview and
sample-data explanations do not account for it. The source investigation still
has no confirmed mechanism for transmitting the saved participant list between
devices.

The user authorised use of the existing password, allowing authenticated
inspection. The password was not saved in source or reports. No password reset
or production configuration change was made.

### Authenticated live checks

- Production serves `index-C4BsKgZG.js`, with SHA-256
  `8a3c10c00dd0fc882839568f6622589c7204284d2f51b27feda4c5481cb447df`.
  A clean build of the saved release produced exactly the same bytes. Neither
  reported participant name appears in the script.
- The earlier `equidistant-london.btjones-me.chatgpt.site` sharing address
  currently serves the same script. The existing project feedback record shows
  plain links to that address on 10 and 16 July and to `equidistant.me` on
  17 July, without participant data in query parameters or fragments. These
  dates do not establish when the reported incident happened.
- Authenticated HTML is a static app shell marked `no-store`. The CDN does
  report static-content cache hits, but the captured HTML and script contain
  no personalised participant state.
- Two independent browser profiles retained their own synthetic participant
  names after edits and reloads, in both directions. A further two-way test
  between Chrome and native desktop Safari also retained separate names.
- All changed names were restored. Existing locations were preserved. No paid
  recommendation or TravelTime calls were made in the live browser tests.

The current build mismatch hypothesis is ruled out for the captured script.
The reported iPhone disclosure is still unresolved: these tests did not
reproduce the affected phone session or establish the historical transfer
mechanism. The remaining evidence is whether that phone still holds the
affected state and the approximate incident date. Existing browser data should
be preserved.

## Evening comparison

The existing TravelTime key returned HTTP 200 for a regular matrix request with
a Wednesday 9 September 2026, 18:00 Europe/London departure. The isolated probe
in `scripts/probe_evening.py` compares 08:30 and 18:00 departures on that date,
using the regular endpoint for both times to avoid confusing time-of-day changes
with differences between the Fast and regular APIs.

The cap is 24 searches in six requests: 12 origins near named neighbourhoods and
96 destinations sampled from existing coverage (64 original, 32 expanded).
Unreachable journeys are tracked separately; missing responses cause a failure
instead of being treated as unreachable. Results are separate from training
data and the shipped model. Running the script without `--execute` only writes
the plan. No retraining or atlas replacement is part of this change.

All six batches completed successfully: 24 searches, 2,304 travel-time values,
and 1,152 paired routes. One additional one-route credential check preceded
the probe. All sampled routes were reachable at both times.

| Measurement | Observed result |
| --- | ---: |
| Average evening minus morning time | +0.18 minutes |
| Median absolute change | 2.32 minutes |
| 90th percentile absolute change | 7.92 minutes |
| Routes changing by more than 5 minutes | 22.48% |
| Routes changing by more than 10 minutes | 5.47% |

For six illustrative groups of two to six people, three changed their best
meeting location among the 96 sampled destinations under the app's balanced
objective (mean travel time plus half the sample standard deviation). For the
West Hampstead/Brixton pair, the chosen sampled location moved 3.2 km and average
evening travel time improved from 38.76 to 33.52 minutes. The three-person
West Hampstead/De Beauvoir/Putney group retained the same sampled choice.

This suggests that an evening model could improve particular routes and groups,
despite the near-zero overall signed average. These measurements cover one
Wednesday and a small set of origins and destinations, not the full destination
map or weekday variation. They compare regular-API 08:30 departures with 18:00
departures, not the accuracy of the current Fast-trained atlas. An evening
retraining decision should follow a larger multi-day validation later.

The plan, raw responses, paired routes, summary, and additional per-origin/group
analysis are retained under `artifacts/runs/evening_preliminary_20260909/` locally.

## Release scope

The follow-up Google sign-in, account permissions, and rolling 60-minute limit
are now implemented locally; see `2026-09-08-google-auth-rollout.md` for their
configuration and rollout status. The initial validation figures below describe
the earlier Luna/cache patch. No changes have yet been published.

Validation: all 22 original/new frontend tests and 22 Worker tests passed,
followed by the updated four-test state suite including an additional storage
isolation test (45 distinct tests in total). The production build passed.
The probe's dry run, formatting/import checks, and patch whitespace checks
passed. No Python model or atlas source was modified.

Reference: [TravelTime departure-search API](https://docs.traveltime.com/api/reference/travel-time-distance-matrix),
[OpenAI Luna capabilities](https://developers.openai.com/api/docs/models/gpt-5.6-luna).
