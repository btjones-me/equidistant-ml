# Future personalised recommendations

## Goal

Improve meeting-place recommendations by finding venues that work for the
whole group, while keeping preference data understandable, optional, and easy
to remove.

## Account model

- Add individual accounts only when the product needs cross-device history.
- Keep a meeting shareable without requiring every participant to register.
- Let a signed-in organiser invite friends to contribute preferences for one
  meeting without exposing their full profile.

## Preference schema

- Walking tolerance and accessibility requirements.
- Venue categories, atmosphere, dietary needs, alcohol preferences, and budget.
- Hard exclusions, soft preferences, and an explicit "no preference" state.
- Time-specific context such as weekday, time of day, and desired duration.

## Group ranking

- Retrieve viable nearby candidates first, then score each participant's fit.
- Treat accessibility and dietary requirements as constraints, not averages.
- Optimise the remaining preferences for broad satisfaction and show why each
  result works for the group.
- Let the organiser adjust the balance between consensus, novelty, and travel
  time without exposing private individual scores.

## Privacy and control

- Collect the minimum preference data needed and make every saved field visible.
- Separate persistent profile preferences from one-meeting answers.
- Provide deletion, export, consent withdrawal, and retention controls.
- Never send names or raw profile histories to search providers when aggregate
  constraints are sufficient.

## Suggested delivery sequence

1. Add anonymous one-meeting preference cards and evaluate recommendation lift.
2. Add optional accounts and encrypted preference storage.
3. Add group invites, consent states, and preference conflict explanations.
4. Add learned preference ranking only after explicit feedback data is useful
   and bias, privacy, and cold-start behaviour have been evaluated.
