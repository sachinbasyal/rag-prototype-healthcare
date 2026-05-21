from __future__ import annotations
import os
import re
import json
import time
from typing import Any, Dict, List
import pandas as pd

def ensure_dir(path:str) -> None:
  os.makedirs(path, exist_ok=True)
  
def now_ms() -> int:
  return int(time.time()*1000)

def clean_text(text:str) -> str:
  if not isinstance(text, str):
    return ""
  text = text.replace("\x00", " ")
  return text

def save_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        
def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
      
def append_csv(path: str, row: Dict[str, Any]) -> None:
    df = pd.DataFrame([row])
    if os.path.exists(path):
        df.to_csv(path, mode="a", index=False, header=False)
    else:
        df.to_csv(path, mode="w", index=False, header=True)