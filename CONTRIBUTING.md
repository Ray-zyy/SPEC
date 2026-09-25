# Contributing

Contributions that improve reproducibility, testing, documentation, portability, or diagnostic coverage are welcome.

Before opening a pull request:

```bash
python -m pytest tests/test_core.py -q
```

Please avoid committing raw hydrological datasets, model checkpoints, private paths, credentials, or large generated artifacts. New numerical methods should include a focused unit test and document any change to the ranking, zero-flow, or minimum-segment conventions.
