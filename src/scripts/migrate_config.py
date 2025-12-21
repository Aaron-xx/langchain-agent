"""Simple JSON to TOML migration tool"""
import json
import sys
from pathlib import Path

def convert_json_to_toml(json_path: str, toml_path: str = None):
    """Convert JSON config to TOML format"""
    try:
        import toml
    except ImportError:
        print("Error: toml package not installed. Run: pip install toml")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    output_path = toml_path or json_path.replace('.json', '.toml')
    with open(output_path, 'w', encoding='utf-8') as f:
        toml.dump(data, f)

    print(f"Migrated: {json_path} -> {output_path}")

if __name__ == "__main__":
    convert_json_to_toml(sys.argv[1] if len(sys.argv) > 1 else "config.json")