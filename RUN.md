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

# Unified classifier - trains on ALL three datasets simultaneously
python main.py -c unified -a train                      # Train all 6 models
python main.py -c unified -a train -m EfficientNetB0    # Train only EfficientNetB0
python main.py -c unified -a train -m ResNet50          # Train only ResNet50  
python main.py -c unified -a train -m VGG16             # Train only VGG16
python main.py -c unified -a train -m InceptionV3       # Train only InceptionV3
python main.py -c unified -a train -m DenseNet121       # Train only DenseNet121
python main.py -c unified -a train -m Xception          # Train only Xception

python main.py -c unified -a analyze
python main.py -c unified -a use