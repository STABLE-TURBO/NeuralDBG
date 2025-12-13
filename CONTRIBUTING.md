# Hey! Welcome to Neural DSL 👋

So you want to contribute? That's awesome! We're genuinely excited to have you here.

Look, we know contributing to open source can feel intimidating (trust us, we've all been there). Maybe this is your first contribution ever, or maybe you're a seasoned pro. Either way, you're in the right place. We're here to help, not to judge.

## Your First Contribution? Perfect!

A few months ago, someone named Alex opened their very first PR. They were nervous—they told us later they rewrote their 10-line bug fix like five times before hitting submit. You know what? The fix was great, and now they're a regular contributor. Everyone starts somewhere.

If you're new to this:
- **It's okay to ask questions.** Seriously, ask away.
- **Small contributions matter.** Fixed a typo? That's a win. Updated a docstring? You're helping people.
- **Mistakes happen.** We've all pushed broken code. We've all forgotten to run tests. It's part of learning.

One contributor once accidentally deleted an entire function in their PR. We laughed about it together, fixed it in like 2 minutes, and moved on. No drama.

## Getting Started (The Fun Part)

Let's get you set up:

```bash
# Grab the code
git clone https://github.com/your-username/Neural.git
cd Neural

# Make a cozy virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Mac/Linux

# Install the dev stuff
pip install -r requirements-dev.txt
pre-commit install  # This sets up automatic checks (it's actually helpful, not annoying)

# Make sure everything works
python -m pytest tests/ -v
```

If something breaks during setup, don't panic. Seriously happens to everyone. Just open an issue or ask for help.

## What Can You Work On?

Honestly? Whatever interests you! But here are some ideas:

### "I Want to Dive Into Something Cool"

**AI stuff** - We've been improving how Neural understands natural language. Remember when Sarah added support for casual phrases like "throw in a dropout layer"? That was a fun PR. The AI code is in `neural/ai/` and it's surprisingly approachable.

**Examples** - Jake spent a weekend fixing examples that had bit-rotted. He found like 3 bugs in the process that we didn't even know existed. Examples are in `examples/` and honestly, running them and seeing what breaks is kinda satisfying.

**Tests** - Look, nobody *loves* writing tests, but we need them. And actually, writing tests is a great way to understand the codebase without the pressure of implementing new features.

### "I'm New, Give Me Something Easy"

**Documentation** - Found something confusing? You're probably not alone. Fix it! The docs are in `docs/` and we promise we won't nitpick your grammar.

**Bug fixes** - Check the GitHub Issues. Some are labeled "good first issue" which means they're perfect for dipping your toes in.

**Tiny improvements** - Better error messages, clearer variable names, whatever. It all adds up.

## About Versions (Don't Worry, It's Not Scary)

Okay, so we version things. Here's how it works in plain English:

**The numbers**: We use Major.Minor.Patch (like 1.2.3)

**Patch releases (0.0.X)** - Remember when Tom found 15 bugs in the shape propagation code over a month? We fixed them all and released 0.2.1 → 0.2.2. That's a patch. Just bug fixes, nothing breaks.

**Minor releases (0.X.0)** - When Maria added the whole hyperparameter optimization feature, that was 0.2.0 → 0.3.0. New cool stuff, but your old code still works.

**Major releases (X.0.0)** - This is the dream: when we have zero known bugs and everything is solid. We'll get there! This might include breaking changes, but we'll be super clear about it.

The cool part? You don't really need to think about this. Just work on your feature or fix, and we'll figure out the version number when we release. There's even automation for it:

```bash
# If you're a maintainer releasing stuff:
python scripts/automation/master_automation.py --release --version-type minor
```

But honestly, as a contributor, you can ignore this whole section. We just wanted to explain it in case you were curious.

## The Actual Workflow

Let's say you found a bug or want to add something:

**1. Branch off**
```bash
git checkout -b fix-that-annoying-thing
```
Name it whatever makes sense to you. We've seen branches called "why-is-this-broken" and honestly, we felt that.

**2. Do your thing**

Write code, add tests (please!), update docs if needed. Work at your own pace.

Pro tip: Commit often. Like, way more often than you think you should. Future you will thank present you.

**3. Make sure it works**
```bash
python -m pytest tests/ -v
python -m ruff check .  # Linting stuff
```

Pre-commit hooks will actually catch a lot of stuff automatically. If they block your commit, don't get frustrated—they're just trying to help.

**4. Push it up**
```bash
git add .
git commit -m "Fix that annoying thing that was driving me crazy"
git push origin fix-that-annoying-thing
```

Then head to GitHub and open a PR. In the description, just explain what you did and why. Doesn't need to be formal—"This was broken, now it's not" is totally fine.

## Code Style (Keep It Reasonable)

We follow PEP 8 mostly because the linters do. Some things we care about:

- **Type hints are nice** - They help catch bugs and make the code clearer. But if you're not sure, that's okay.
- **Docstrings are helpful** - Especially for complex functions. Doesn't need to be a novel.
- **Small functions** - If you can break it up, do. But don't stress about it.

The linters (ruff, pylint) will keep you honest. Sometimes they're annoying, but they catch real issues.

## When Things Go Wrong

**Tests failing?** Yeah, that happens. Run them locally, see what's up. If you're stuck, push your branch anyway and we can help troubleshoot in the PR.

**Linter complaining?** Most of the time it's just formatting. Pre-commit can auto-fix a lot of it. If it's being unreasonable, let us know.

**Not sure if your approach is right?** Open a draft PR early! Seriously, we love seeing works-in-progress. It's way easier to course-correct early than after you've written 500 lines.

**Broke something accidentally?** We've all been there. Just tell us. We'll help fix it. No judgment.

## Some Real Stories

**The Time Someone Added HPO** - Maria wanted to add Optuna integration for hyperparameter tuning. She wasn't sure if it should be core or optional. We talked through it, decided to make it optional (so it doesn't bloat the base install), and now tons of people use it. The discussion was valuable.

**The Accidentally Genius Bug Report** - Someone filed a bug report that was basically "this doesn't work idk why." But they included their example file, and it helped us discover a whole class of edge cases we'd missed. Bug reports don't need to be perfect.

**The Documentation Hero** - Chris spent a weekend just reading docs and fixing confusing parts. No code changes, just clarity. We got like 3 messages that week saying "wow, the docs make sense now." Never underestimate documentation.

**The "Oops I Pushed to Main" Incident** - Yeah, that happened once (okay, twice). We fixed it, laughed about it, and tightened up the branch protections. We're all human.

## Dependencies (If You're Adding Libraries)

Adding a new library? Cool! Just think about where it fits:

- **Core** (`CORE_DEPS` in setup.py) - Only if Neural literally cannot function without it. This is rare.
- **Optional feature groups** - Most new stuff goes here. ML frameworks, visualization tools, cloud integrations—they're all optional so users don't have to install everything.

When Emily added Dash for the dashboard, she made it optional with `pip install neural-dsl[dashboard]`. Perfect. Users who want it can get it, others don't pay the install cost.

Quick checklist:
- Does it play nice with Python 3.8+?
- Is the license okay? (MIT-compatible is great)
- Can we live without it in the core package?
- Is it maintained?

And definitely test that core stuff still works without your new dependency:
```bash
python -m venv test_minimal
test_minimal\Scripts\activate
pip install -e .  # Just core
# Try basic commands
```

## What We're Focused On

If you want to work on something high-impact, check out the priorities:

1. **Experiment tracking** - People really want this. Think MLflow integration.
2. **Data pipelines** - Connecting to real data sources.
3. **Model deployment** - Getting models into production.
4. **Making things faster** - Performance always matters.
5. **Model versioning** - Tracking model evolution.

But honestly, if you're passionate about something else, go for it! Passion projects often turn into the best features.

## Getting Help

**Stuck? Confused? Not sure if your idea makes sense?**

- **GitHub Issues** - Ask questions, no such thing as a dumb question
- **Pull Requests** - Open a draft PR and tag us, we'll help
- **Discussions** - Great for "hey, what do you think about..." conversations
- **Discord** - Real-time chat (link's in the README)

We're friendly, we promise. The whole point is to build something cool together.

## A Note on Mistakes

You're going to make them. We all do. Here's what's actually important:

- **We're all learning.** Every single one of us is figuring things out as we go.
- **Mistakes are how we improve the project.** That bug you found? Now we can fix it. That edge case you discovered? Now we can handle it.
- **Communication beats perfection.** We'd rather have a messy PR with good communication than radio silence because you're worried it's not perfect.

Last month someone submitted a PR that accidentally broke like 6 tests. You know what happened? We helped them fix it, and in the process, realized our error messages were terrible. So we improved those too. Everyone wins.

## The Part Where We Say Thanks

Real talk: open source runs on people giving their time because they care. Whether you fix a typo, add a feature, or just report a bug—it matters. You're making Neural DSL better for everyone who uses it.

We appreciate you being here. Seriously.

Now go make something cool! 🚀

---

**P.S.** - If you read this whole thing, you're probably overthinking it. Just start somewhere, and we'll figure it out together. You've got this!
