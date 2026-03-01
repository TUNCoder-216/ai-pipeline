import os
import sys
import logging

# 1. FORCE PATH: This ensures the script can see the 'src' folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 2. ENABLE LOGGING: Without this, your model's log.info() calls are invisible
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from src.model import SentimentModel

model = SentimentModel()
model.load()  # downloads ~250MB first time, cached after

result = model.predict("I absolutely love this product!")
print(result)
# Expected: {'label': 'POSITIVE', 'score': 0.9987, ...}

result2 = model.predict("This is the worst thing I ever bought.")
print(result2)
# Expected: {'label': 'NEGATIVE', 'score': 0.9994, ...}

batch = model.predict(["Amazing!", "Terrible.", "It is fine."])
print(batch)
if __name__ == "__main__":
    main()