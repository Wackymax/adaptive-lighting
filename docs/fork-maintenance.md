---
icon: lucide/git-branch
---

# Fork ownership and upstream synchronization

## Ownership

This distribution is maintained in the [Wackymax/adaptive-lighting
fork](https://github.com/Wackymax/adaptive-lighting). It is a fork of the
[upstream basnijholt/adaptive-lighting project](https://github.com/basnijholt/adaptive-lighting).

The upstream project, its authors, its license, and its existing contributors
remain part of the history and attribution of this fork. Fork-specific
documentation and changes should be clearly identified as such; the fork must
not imply that upstream has adopted or supports fork-only behavior.

The primary places for fork issues, releases, and installation links use
Wackymax. Historical links to upstream issues, assets, or contributor commits
are retained where they identify the original source of that work.

## Synchronization workflow

Keep the upstream remote separate from the fork remote and inspect changes
before merging them:

```bash
git remote add upstream https://github.com/basnijholt/adaptive-lighting.git
git fetch --prune upstream
git log --oneline --decorate HEAD..upstream/main
git diff --stat HEAD...upstream/main
```

After review, synchronize the fork branch using the repository's normal branch
policy. A merge preserves the fork history explicitly:

```bash
git merge --no-ff upstream/main
```

If the branch policy uses rebases instead, rebase only after confirming that
fork-only documentation, metadata, and future intelligence work are preserved.
After either path:

1. review the complete diff, including generated documentation sections;
2. rerun the focused documentation and formatting checks;
3. verify that the manifest and package versions still match; and
4. push to the Wackymax fork only after the fork-specific changes are
   intentional.

Do not copy upstream credentials, private deployment details, or local
Home Assistant state into the fork while synchronizing.

## Fork versioning

The upstream baseline for this branch is `1.31.0`. The fork uses the
HACS-friendly prerelease `1.31.0b1` in both `pyproject.toml` and
`custom_components/adaptive_lighting/manifest.json`. Keep these values aligned.

Increment the prerelease for subsequent fork-only iterations, or move to the
next upstream base version when the fork is synchronized with it. Do not label
fork-only behavior as a stable upstream release, and do not use an arbitrary
version suffix that one of the Python or HACS parsers cannot compare.

## Documentation ownership rules

- User-facing repository, issue, release, and HACS links should point to
  Wackymax when the resource exists in the fork.
- Upstream links should remain when they are needed for author attribution,
  historical issue context, original media, or the source project.
- Architecture documents must distinguish implemented behavior from planned
  activation and must not introduce unsupported configuration keys.
- Toothless examples must contain entity-level examples only; never add
  credentials, tokens, private hostnames, or private network addresses.
