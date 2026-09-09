# Google accounts and access controls

Implementation replaces the shared-password login with Google OpenID Connect.
It does not claim to resolve the reported historical iPhone data disclosure.

## User-visible behavior

- Anyone with a Google account can sign in. Only the server-managed email
  allowlist can open diagnostics, inspect usage aggregates, or request paid
  TravelTime comparisons.
- Recommendations use GPT-5.6 Luna. Other accounts get five uncached research
  attempts in any rolling 60-minute window across sessions and devices.
  Cached results remain available. Failed provider attempts consume a slot.
- Approved accounts bypass the personal quota, while all existing global
  recommendation, photo, and TravelTime cost caps remain active.
- Participant workspaces remain on the device, separately for each Google
  account. Legacy unowned browser data is preserved, but not automatically
  adopted by whichever account signs in first. No cross-device workspace
  synchronization has been introduced.
- Sign-out revokes the server session. Account changes in another browser tab
  revalidate the visible workspace; stale API requests are rejected.

## Implementation and checks

OAuth uses a browser-bound, expiring, one-use state record, PKCE, and a nonce.
Google ID tokens are verified with `jose` against Google's signing keys,
issuer, audience, expiry and verified email. The account key derives from
Google's stable subject, not its display name or a client-supplied email.
Opaque session tokens are stored only as hashes in D1, expire after seven days,
and use secure HTTP-only cookies. The old password cookie grants no access.

The rolling quota uses one conditional SQLite write before paid work starts.
Queries and recommendation caches are scoped to the authenticated account.
Photo URLs are signed for that account, the exact photo and an expiry; direct
arbitrary photo requests require privileged access. Same-origin checks protect
state-changing requests. Identity-bearing responses use private/no-store.
Sign-in starts are bounded per hashed client address, without storing raw IPs.

The account and sign-in migrations are additive. Existing analytics, caches,
password configuration and user browser storage are preserved. The old password
setting remains only to support a rollback to the old release; the new code
never reads it. Sites must apply both generated migrations before Worker upload.

Validation uses real SQLite with a D1 adapter, cryptographically signed test
Google tokens, mocked external providers, and frontend account-switching tests.
It covers forged/expired/revoked sessions, callback replay and missing browser
state, cross-origin requests, server allowlisting, cross-hour and simultaneous
quota requests, signed photos, and account-isolated workspaces.

## Configuration and rollout status

The approved three-person allowlist, canonical origin, and newly generated
signing secret are saved as hosted environment revision 8. Private values are
not in source. Existing provider keys are preserved.

Google's existing project `starlit-surge-296319` now has an Equidistant web
sign-in client. The user completed the consent-policy approval. The authorized
redirect is exactly `https://equidistant.me/auth/google/callback`. Google is
configured for an external production audience, with `equidistant.me` as the
authorized domain and `/privacy` as its public privacy notice. Only
`openid email profile` is requested. Client credentials are stored in hosted
environment revision 9; the temporary credential download was removed.

The release is prepared for the existing public Site at `https://equidistant.me`.
Publication and the real Google-login check are the final rollout steps.

References: [Google OpenID Connect](https://developers.google.com/identity/openid-connect/openid-connect),
[Google sign-in branding](https://developers.google.com/identity/branding-guidelines),
[jose](https://github.com/panva/jose).

Current validation: 26 frontend tests and 32 Worker/authentication tests passed
(58 total). The production build and deployment archive checks passed. A dependency audit found four existing
advisory-bearing packages in the frontend build toolchain (Browserslist,
baseline-browser-mapping, Nano ID, and PostCSS); the new authentication runtime
library had no reported advisory. These build-tool updates remain separate from
this authentication rollout.
