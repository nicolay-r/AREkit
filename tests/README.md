## Unit Testing

Install test required dependencies:
```bash
pip install -r dependencies.txt
```

Performing a quick `AREkit` library installation
[[link]](https://stackoverflow.com/questions/19048732/python-setup-py-develop-vs-install)
```bash
pip install -e ../ --no-deps
```

Using `pytest` to run all the test and gather report into `pytest_report.html` document.
```bash
python -m pytest --html=pytest_report.html --self-contained-html --continue-on-collection-errors .
```
