"""AOI definitions (bounding boxes)

Add AOIs here as needed. Each AOI is a dict with keys:
  - min_lon, min_lat, max_lon, max_lat

The app imports this module to populate the AOI dropdown.
"""

AOIS = {
    "Afedena": {
        "min_lon": 39.290620,
        "min_lat": 13.667650,
        "max_lon": 39.293380,
        "max_lat": 13.670350,
        "notes": "Approximate AOI provided by user"
    },
    "Bale Mountains": {
        "min_lon": 39.0,
        "min_lat": 8.2,
        "max_lon": 39.8,
        "max_lat": 8.8,
        "notes": "Bale Mountains National Park"
    },
    "Simien Mountains": {
        "min_lon": 38.0,
        "min_lat": 13.0,
        "max_lon": 38.8,
        "max_lat": 13.5,
        "notes": "Simien Mountains National Park"
    },
    "Awash National Park": {
        "min_lon": 39.8,
        "min_lat": 8.5,
        "max_lon": 40.5,
        "max_lat": 9.2,
        "notes": "Awash National Park"
    }
}

def get_aoi_list():
    """Return list of AOI names."""
    return list(AOIS.keys())

def get_aoi_bbox(name):
    """Return bounding box dict for AOI name, or None if not found."""
    return AOIS.get(name)