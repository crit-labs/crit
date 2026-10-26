# crit-eval

crit-eval, crit's common speed, quality and exactness suite, lives in its own repository:
**[crit-labs/crit-eval](https://github.com/crit-labs/crit-eval)**.

crit installs it as a dependency and runs it as `crit eval` (`crit-eval` on its own). For a checkout, clone
crit-eval beside crit and install both editable:

```sh
python -m pip install -e core/python -e ../crit-eval -e .
```

- [What crit-eval measures](https://github.com/crit-labs/crit-eval#what-eval-measures)
- [Guide](https://github.com/crit-labs/crit-eval/blob/main/docs/README.md)
- [Limitations](https://github.com/crit-labs/crit-eval#limitations)
