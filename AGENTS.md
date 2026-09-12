# Equidistant agent instructions

## Source of truth

- The user's personal GitHub repository, `btjones-me/equidistant-ml`, is the
  canonical source repository. Keep `origin` pointed at
  `git@github.com:btjones-me/equidistant-ml.git`.
- Sites uses a separate source repository for deployment. Publishing to Sites
  does not update GitHub. The existing Site is identified by
  `frontend/.openai/hosting.json`; production is `https://equidistant.me`.

## Keep GitHub and releases in sync

The user has requested that completed project changes be kept in their personal
GitHub repository. Treat this as standing authorization to commit and push
completed work within the requested task. Respect any later instruction to keep
work local, leave it uncommitted, or prepare a review without merging.

1. Fetch the relevant GitHub branches before integration. Inspect the working
   tree and remote history, preserve unrelated work, and never force-push over
   remote changes. Use `codex/` for new working branches.
2. Validate changes appropriately, review the files being committed, and push
   completed work to GitHub before declaring the task finished. For completed
   changes intended for the main project, update `origin/main` when a safe
   fast-forward is possible. Follow branch protections or an explicitly
   requested pull-request workflow; report pending review or merge clearly.
3. For an authorized Sites release, first ensure the exact validated release
   commit is on GitHub `main`. Push that same commit to the existing Sites
   source repository, build/package from that source, and save/deploy that exact
   SHA using the Sites workflow. Keep Sites credentials out of Git remote URLs
   and configuration; never replace `origin` with the Sites repository.
4. Verify the GitHub remote branch after pushing and the deployment status after
   publishing. The deployed commit must be present in GitHub's history. Push any
   subsequent release notes or documentation updates to GitHub too.
5. Repository or documentation maintenance alone does not require a new Sites
   deployment. When application changes are intentionally not deployed, say so.
   If a push, merge, or deployment fails, report the exact remaining sync gap;
   do not describe local commits or a successful Sites deploy as GitHub sync.

## Public repository hygiene

This GitHub repository is public. Keep credentials, OAuth client secrets,
session tokens, personal allowlists, and private user data out of commits.
Use ignored local environment files and Sites runtime environment settings for
secrets. Inspect the complete outgoing commit range before pushing, not just the
last commit or current working-tree diff.
