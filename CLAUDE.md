# CGP Subnet Development Guide

## Build & Test Commands
- Run all tests: `python3 -m pytest -s --disable-warnings tests/test_*`
- Run specific test: `python3 -m pytest -s tests/test_file.py::test_function`
- Install requirements: `pip install -r requirements.txt -r requirements_test.txt`
- Run miner: `python neurons/miner.py`
- Run validator: `python neurons/validator.py`

## Code Style Guidelines
- **Imports**: Standard library first, then third-party, then local modules (alphabetically sorted)
- **Naming**: 
  - Classes: PascalCase (e.g., `BaseNeuron`)
  - Functions/variables: snake_case (e.g., `check_registered`)
  - Constants: UPPER_CASE
- **Types**: Use type hints for function parameters and return values
- **Documentation**: Triple-quoted docstrings with clear descriptions
- **Error Handling**: Use explicit logging with appropriate exit codes
- **Configuration**: Use the centralized config system in utils/config.py
- **Logging**: Utilize the custom logging setup from utils/logging.py