# Public London landing page

The anonymous homepage now describes the London product and presents a playful
hexagon map. Pointer movement highlights cells; clicking/tapping produces a
ripple; neighbourhood buttons provide keyboard interaction. Reduced-motion
preferences suppress the ripple animation. The illustration contains only
static coverage geometry and hard-coded public neighbourhood labels, not model
predictions. It makes no network or browser-storage calls.

Actual planning, model assets, geocoding, venue research and account data retain
Google authentication. Only the exact decorative script, robots/sitemap and
existing public branding images are additionally exposed. Public static files
can be cached for five minutes. Account-dependent responses remain private and
uncacheable. Unknown anonymous pages return 404 rather than duplicate 200 pages.

The homepage includes a London-specific title, description, canonical URL,
existing social preview, explanatory HTML and FAQ. No live-route verification,
sharing, group synchronization or venue-result redesign is included.

## SEO assessment

Competitors have stronger public content structures than the previous sign-in
screen: [togather](https://meettogather.com/) links city and situation guides;
[Halfway London](https://halfwaylondon.com/) links halfway and meeting-place
pages; [Equidistance](https://equidistance.io/londons-most-equidistant-pub/)
publishes original research with a methodology and data. These are observations
of crawlable content, not verified traffic or ranking comparisons.

This release establishes a public indexable entry point while retaining the
login boundary. Further location-specific content should answer real questions
and use evidence, not mass-produced doorway pages. Search Console submission and
indexing outcomes are not part of this release.

## API protection assessment

The existing Worker reads the OpenAI key from its runtime environment and adds
it to the server-to-server request header. It is not supplied to the browser or
model prompt by this code. Google login is helpful but cannot by itself prevent
abuse by signed-in users or protect a key leaked elsewhere.

Current source caps ordinary accounts at five uncached research requests per
rolling hour, with global research caps of 30/day and 300/month. Approved accounts
bypass only the personal cap. Photo and live-comparison paths have separate
limits. Database failure blocks paid requests, and concurrency tests check that
personal reservations do not exceed the limit. These are application request
caps, not a guarantee of a maximum currency amount or protection against every
availability attack. Multiple accounts can exhaust the shared allowance.

Remaining recommendations: independently verify provider-side key scope and
usage controls, alert on unusual usage, and review edge rate limiting when
launch traffic warrants it. Provider account settings and edge rules were not
changed or fully audited here. No credential was rotated or printed.

## Validation

Frontend and Worker suites include public-route isolation, zero upstream/D1
calls for the anonymous landing, no network operations in decorative code,
authenticated-root preservation, HTTP methods, metadata and crawl endpoints.
Use `node build/landing-geometry.mjs` to regenerate the committed decorative
geometry from the atlas when intentionally changing its coverage.

## Release verification

Release commit `330153ea8e4156aa01e249cfa28b2dc8672e7d02` was pushed to
GitHub main and the existing Sites source. Sites version 23 deployed successfully
on 13 September 2026 with the existing environment revision 9. Deployment ID:
`appgdep_6aa5eb564a6481919150663d818fdef4`.

All 26 frontend and 35 Worker tests passed. The release build and archive checks
passed. Independent anonymous requests to `https://equidistant.me` verified the
new homepage, decorative script, robots, sitemap and existing social image
returned 200; session and venue APIs returned 401; the atlas returned a 404
sign-in response. The site's edge filter rejected Python's default user agent
with 403/1010; the same checks with a standard browser user agent succeeded.
No paid provider calls or account sign-in were made during live checks.
This is HTTP and automated verification, not a browser visual/interaction audit
or a complete security assessment.
