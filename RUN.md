### 1. Just Run once for project setup
```bash
bash ./setup.sh
```

### 2. Activate Environment (Every time open vs code run this script)
```bash
source ./venv/Scripts/activate
```

### 3. Main Command to train and Analyze model.
```bash
python main.py -c fingerprint -a train 
python main.py -c fingerprint -a analyze

python main.py -c face -a train
python main.py -c face -a analyze

python main.py -c iris -a train
python main.py -c iris -a analyze