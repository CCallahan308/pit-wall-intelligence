# Contributing

This is primarily a portfolio project, but PRs that fix bugs, improve modeling, or extend the dashboard are welcome.

## Dev setup

```bash
git clone https://github.com/CCallahan308/pit-wall-intelligence.git
cd pit-wall-intelligence
uv sync --extra dev
uv run pre-commit install  # optional
```

## Before opening a PR

```bash
make lint
make test
```

CI runs `ruff check`, `pytest`, and `dbt test`. Schema drift fails the build.

## Modeling changes

Any change to the degradation model or undercut classifier must:

1. Update the corresponding model card in `docs/model_cards/`
2. Re-run `notebooks/03_model_validation.ipynb` and check in the updated metrics
3. Note the metric delta in the PR description
