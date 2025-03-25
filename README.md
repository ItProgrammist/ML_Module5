**Full Name**: [Дмитрий Козлов]
**Group**: [972301]

## How To

### 1. **Install Poetry and Project Dependencies**
To install all project dependencies, make sure you have Poetry installed. If Poetry is not already installed, run the command:


```bash
curl -sSL https://install.python-poetry.org | python3 -

poetry install
```
(then follow installation instructions)

### **2. Run the Main file**

poetry run python src/model.py

train
```bash
poetry run python src/model.py train --dataset="data/train.csv"
```

predict
```bash
poetry run python src/model.py predict --dataset="data/test.csv"
```
