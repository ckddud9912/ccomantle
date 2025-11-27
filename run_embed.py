import os
import json

# 이미 생성되었는지 확인
if os.path.exists("embedding_dictionary.json"):
    print("[INFO] embedding_dictionary.json already exists. Skipping.")
else:
    print("[INFO] embedding_dictionary.json not found. Running embedding_precompute.py...")
    os.system("python embedding_precompute.py")

print("[INFO] run_embed.py finished.")
