"""
storage.py — Handles file operations for saving and loading baselines and logging session data.
"""
import json
import os
from typing import List, Dict, Any

class BaselineStorage:
    """
    Handles serialisation and loading of baseline specifications from disk.
    """
    @staticmethod
    def load(path: str) -> Dict[str, Any]:
        """
        Load a baseline specification from a JSON file.
        
        Args:
            path: Absolute path to the baseline file.
            
        Returns:
            A dictionary containing name, description, and list of examples.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Baseline file not found at: {path}")
            
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return {
                "name": data.get("name", "default"),
                "description": data.get("description", ""),
                "examples": data.get("examples", [])
            }

    @staticmethod
    def save(path: str, name: str, examples: List[str], description: str = "") -> None:
        """
        Save a baseline specification to a JSON file.
        
        Args:
            path: Absolute path to the destination file.
            name: The name of the baseline category/topic.
            examples: A list of known-good responses.
            description: Optional text describing the baseline focus.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        data = {
            "name": name,
            "description": description,
            "examples": examples
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


class SessionLogger:
    """
    Appends session turn scores and verdicts to log files for offline analysis.
    """
    def __init__(self, log_dir: str):
        """
        Initialise the session logger.
        
        Args:
            log_dir: Directory where log files should be written.
        """
        self.log_dir = log_dir

    def log_turn(self, session_name: str, result: Dict[str, Any]) -> str:
        """
        Append a turn verdict metric log entry to a JSON Lines file.
        
        Args:
            session_name: The name of the session or baseline category.
            result: The dictionary representation of a DriftVerdict.
            
        Returns:
            The path of the file that was logged to.
        """
        os.makedirs(self.log_dir, exist_ok=True)
        log_file = os.path.join(self.log_dir, f"drift_metrics_{session_name}.jsonl")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")
        return log_file
