# Contributing

Contributions are welcome. This guide is maintained by practitioners — if something is wrong, outdated, or missing, open a PR.

## What makes a good contribution

**Fix factual errors.** The field moves fast. If a tool version, API, or pattern is stale, update it.

**Add production war stories.** The best content in this guide came from real incidents and real deployments. If you have a "here's what we learned the hard way" story, that's exactly what belongs here.

**Improve the diagrams.** ASCII diagrams that are clearer or more accurate are always welcome.

**New labs.** If you built something that maps to a phase and would make a good hands-on exercise, propose it.

## What doesn't belong here

- Marketing content or vendor comparisons that read like ads
- Content targeting ML engineers or data scientists (this guide is for infrastructure engineers)
- Explanations of ML theory beyond what's needed to understand the infrastructure implications

## How to contribute

1. Fork the repo
2. Create a branch: `git checkout -b fix/dcgm-metric-names`
3. Make your change
4. Open a PR with a clear description of what changed and why

## PR conventions

- Keep PRs focused — one thing per PR
- If you're updating a config, test it first
- Reference the phase or section your change relates to in the PR description

## Issue templates

Use the issue templates for:
- [Suggesting a resource](./.github/ISSUE_TEMPLATE/resource-suggestion.md)
- [Reporting an error or outdated content](./.github/ISSUE_TEMPLATE/lab-feedback.md)
