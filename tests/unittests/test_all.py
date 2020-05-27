import os

excludes = {"test_all.py", "test_clf_datasets.py", "test_reg_datasets.py"}

for file in sorted(os.listdir()):
    if file not in excludes:
        os.system(f"python {file}")
