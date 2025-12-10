from pathlib import Path
import json
import ast

def load_court_config(court_number: int, config_path: str = "court_info/court_exclusion_zones.json") -> dict:
    with open(config_path, 'r') as f:
        data = json.load(f)
    
    court_key = str(court_number)
    if court_key not in data['courts']:
        raise KeyError(f"Court {court_number} not found in config")
    
    court_data = data['courts'][court_key]
    zones = court_data['exclusion_zones']
    
    def parse_coords(coord_list: list) -> list:
        """Convert list of string coords to list of tuples."""
        parsed = []
        for coord_str in coord_list:
            # Parse string "(x, y)" to tuple (x, y)
            # ast.literal_eval safely evaluates the string as a Python literal
            coord_tuple = ast.literal_eval(coord_str)
            parsed.append(coord_tuple)
        return parsed
    
    result = {
        'court_number': court_number,
        'LEFT_EXCLUSION_ZONE': parse_coords(zones.get('LEFT_EXCLUSION_ZONE', [])),
        'RIGHT_EXCLUSION_ZONE': parse_coords(zones.get('RIGHT_EXCLUSION_ZONE', [])),
        'COURT_POLYGON': parse_coords(zones.get('COURT_POLYGON', [])),
    }
    
    print(f"  Court number: {result['court_number']}")
    print(f"  Left zone: {len(result['LEFT_EXCLUSION_ZONE'])} points")
    print(f"  Right zone: {len(result['RIGHT_EXCLUSION_ZONE'])} points")
    print(f"  Court polygon: {len(result['COURT_POLYGON'])} points")
    
    return result