"""
从当前目录下随机选择一个程序
"""

import os
import numpy as np


files = os.listdir()

file = np.random.choice(files)

# if file.endswith('.py') and file is not 'test.py' and file is not "random_choice.py":
#     print(file)
print(file)