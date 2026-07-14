# Browser Atlas Coverage Expansion

## Current limit

The deployed browser atlas supports origins from `51.46..51.56` latitude and
`-0.245..0.03` longitude. Its 3,032 H3 destination cells extend slightly beyond
that to approximately `51.448..51.572` and `-0.257..0.042`. This covers the
existing Zones 1-3 focus, but not Harrow or Bromley.

The trained model used a wider London training box, so a wider atlas can be
generated without immediately retraining. Accuracy outside the current central
holdout is not yet demonstrated, however, and should not be presented as equally
reliable until it is checked against new TravelTime labels.

## Proposed Harrow-to-Bromley atlas

1. Extend destination coverage to the existing `outer_context` box in
   `params.yaml`: north `51.65`, south `51.37`, west `-0.43`, east `0.18`.
2. Keep H3 resolution 9 for Zones 1-3 and important rail corridors. Use
   resolution 8 for the lower-priority outer context, with overlapping buffered
   bands and one owner per H3 cell.
3. Replace the regular origin grid with nested anchors: retain the existing 560
   fine central anchors, add medium-density Zone 3 anchors, then coarser outer
   anchors around Harrow, Bromley, and other major interchanges.
4. Export atlas files in spatial chunks and load only the chunks near each
   participant. Keep the graph surface lazy-loaded. This prevents the initial
   browser download growing from roughly 1.6 MB to tens of megabytes.
5. Fetch a capped TravelTime holdout concentrated on outer rail corridors,
   Thames crossings, and the new boundary. Start with 60-100 origins and a
   stratified destination sample rather than an uncapped all-pairs request.
6. Report central, Zone 3, and outer-context MAE and p90 error separately. Do
   not enable the wider UI by default unless the outer holdout is acceptable.
7. If the existing model fails the outer holdout, add the new labelled origins
   to training, rebuild graph and access features, retrain, and repeat the same
   spatial holdout before exporting the final atlas.

## Acceptance

- No visual gaps at H3 band boundaries.
- Browser interpolation adds no more than 2 minutes MAE over direct model
  inference in each coverage band.
- Outer-context error is explicitly shown in metrics and does not materially
  degrade the existing Zones 1-3 result.
- Initial model download remains below 5 MB, with additional chunks fetched on
  demand and cached by the browser.
