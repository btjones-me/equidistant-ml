# Shared pub availability

The unlisted `/50pubs` planner has a Group availability action. A creator reviews a separate snapshot of up to six starting locations, with one or two people per location. Names separated by `&`, `+`, or the word `and` are suggestions only; the review form is authoritative. People have stable IDs and may later be renamed, removed, or linked/unlinked without changing the personal browser planner.

## Access and storage

Existing Google sign-in remains required. The creator becomes a member. A random 256-bit invitation in the URL fragment can be exchanged for membership only after an explicit Join action. Invitations expire after seven days; only their SHA-256 digests are stored. An early first-party script removes the fragment before other page scripts, retaining it in tab-local storage for at most ten minutes across sign-in. All group reads and writes require session-bound membership and the active-account header. Mutations also require same-origin requests. Original configured presets remain owner-only, including when a stale broad-audience setting exists.

Group records use D1 tables from migration `0003_shared_groups.sql`. Each availability cell is written atomically, with unknown/free/unavailable states. All members may edit any person. Different fields of a person are updated independently to preserve unrelated concurrent edits. Deleted people are soft-deleted and excluded from availability reads/writes. Group membership is not the same as an availability participant: inviting an account does not silently add a person.

The visible month is initialized in Europe/London. Changes refresh every ten seconds while the panel or an applied selection is active and the page is visible. In-flight requests are invalidated by context changes. Failed refreshes clear active recommendations and expose an error rather than leave stale scores visible. A date selects attendees, not model timetables or pub hours. Each shared origin is calculated once, but every available individual contributes once to the scoring objective.

## Product boundaries

At most 20 groups per creator, 24 member accounts, 12 people and two people per origin. All members can generate invitations and edit people/availability. There is no calendar integration, notification system, owner-role hierarchy, self-service membership revocation or permanent group deletion in this small first version. The privacy notice explains server storage and how to request deletion. New people can be linked to the group's existing starting locations or left unlinked; missing locations must be resolved before initially applying a date.

## Verification

Tests cover authenticated route handling, cross-account and cross-group isolation, same-origin writes, invitations, expiry, deleted people, invalid payloads and dates, rate limits, atomic edits, person-weighted scoring, modal interactions, asynchronous saves and race conditions, and preservation of personal workspaces. TypeScript and production build checks are run before release.

Managed browser preview could not start in this environment: Vite failed in `os.networkInterfaces()` with `uv_interface_addresses` / `ERR_SYSTEM_ERROR`. Mobile layout is implemented with responsive CSS and component tests, but real-browser mobile interaction, real Google sign-in invitation exchange, and two-account acceptance were not visually verified. No production group or availability records were created during development.
