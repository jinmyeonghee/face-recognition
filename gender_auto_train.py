## 대용량 데이터셋을 분할 학습 시키기 위한 train_gender1.py ~ train_gender42.py까지의 코드 실행

import subprocess
from glob import glob
import re

# time
from tqdm import tqdm
import time

file_list = glob("./train_gender*.py")  

for f in file_list:
    strat_time = time.time()
    subprocess.call(['python', f])
    end_time = time.time()
    print("prepare time(sec): ", round(end_time - strat_time,1))
    print("------------------------------", f)
    
    