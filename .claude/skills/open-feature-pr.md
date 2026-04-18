# open-feature-pr

Open a properly-labeled, non-draft GitHub PR for the current feature branch.

## Usage

```
/open-feature-pr
```

Run this after all commits are pushed. It completes a feature or milestone.

## Steps

1. Confirm the branch is pushed and has commits ahead of `main`:
   ```bash
   git log main..HEAD --oneline
   ```

2. Determine the milestone ID from the branch name or `.agent-plan.md`.

3. Check for an existing milestone; create one if missing:
   ```bash
   gh api repos/OWNER/REPO/milestones --jq '.[] | {number, title}'
   gh api repos/OWNER/REPO/milestones -X POST -f title="MN" -f description="..." -f state="open"
   ```

4. Identify appropriate labels (`enhancement`, `bugfix`, `tests & testing`, etc.). Create missing labels if needed:
   ```bash
   gh label create "label-name" --color "RRGGBB" --description "..."
   ```

5. Open the PR — **never as a draft**:
   ```bash
   gh pr create \
     --title "type(scope): subject" \
     --label "label1,label2" \
     --milestone "MN" \
     --body "$(cat <<'EOF'
   ## Problem
   <what was broken or missing>

   ## Solution
   <summary of approach>

   ## Changes
   | File | What changed |
   |------|-------------|
   | ... | ... |

   ## Test plan
   - [x] unit tests — N passed
   - [x] full suite — N passed, 0 failed
   - [x] ruff + mypy — clean

   🤖 Generated with [Claude Code](https://claude.com/claude-code)
   EOF
   )"
   ```

6. Confirm the PR URL is returned and the PR is not a draft.

## Notes

- A feature is **not complete** until this step is done — a pushed branch is not a deliverable.
- Always assign a milestone. If the milestone doesn't exist yet, create it.
- The PR body must include a problem statement, solution, per-file change summary, and test results.
