**Full Name**: [Дмитрий Козлов]
**Group**: [972301]

## How To

### 1. **Install Poetry and Project Dependencies**
To install all project dependencies, make sure you have Poetry installed. If Poetry is not already installed, run the command:


```bash
curl -sSL https://install.python-poetry.org | python3 -

poetry install
poetry build
```
(then follow installation instructions)

### **2. Install ClearML** ###

```bash
python3 -m venv clearml-env  # Create virtual environment
source clearml-env/bin/activate  # Activate virtual environment
pip install clearml
nano ~/.zshrc
```

(Or if you use bash)
```bash
nano ~/.bash_profile
```

Add these strings at the end of the file:
```bash
export CLEARML_ACCESS_KEY="your_access_key"
export CLEARML_SECRET_KEY="your_secret_key"
```

Make changes active
```bash
source ~/.zshrc
```

Initialize ClearML Project
```bash
clearml-init
```

### **3. Run the Main file**

train
```bash
poetry run python src/model.py train --dataset="data/train.csv"
```

predict
```bash
poetry run python src/model.py predict --dataset="data/test.csv"
```
