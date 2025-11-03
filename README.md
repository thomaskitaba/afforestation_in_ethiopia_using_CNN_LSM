# afforestation_in_ethiopia_using_CNN_LSM

This repository contains tools to monitor forest growth and reforestation progress using Earth Engine and ML models. The Streamlit app provides dashboard, data analysis, ML predictions, map view and reporting features.

## AOI (Area of Interest) dropdown

The Streamlit app includes an AOI dropdown so you can quickly switch between named monitoring areas. AOIs are defined in `utils/aois.py` as named bounding boxes with the following keys:

- `min_lon`, `min_lat`, `max_lon`, `max_lat`

The app also provides a `Custom` option that lets you manually enter bounding box coordinates.

### Add or edit AOIs

Edit `utils/aois.py` and add entries to the `AOIS` dictionary. Example:

```python
AOIS = {
    "Afedena": {
        "min_lon": 39.290620,
        "min_lat": 13.667650,
        "max_lon": 39.293380,
        "max_lat": 13.670350,
        "notes": "Approximate AOI provided by user"
    }
}
```

After saving, reload the Streamlit app and the new AOI will appear in the dropdown.

## Running the app

Install dependencies listed in `requirements.txt`, then run:

```bash
streamlit run app.py
```

## Notes

- The initial AOI provided in this repo is named `Afedena` and matches the bounding box in the example above.
- If Earth Engine is not available the app uses sample data for demonstration.
# afforestation_in_ethiopia_using_CNN_LSM
<!-- ({'geojson_file': '/mnt/data/afedena_aoi_approx.geojson', 'center_lat': 13.669, 'center_lon': 39.292, 'estimated_area_m2': 69998.24, 'estimated_area_ha': 7.0, 'bounding_box': {'min_lon': 39.29062, 'min_lat': 13.66765, 'max_lon': 39.29338, 'max_lat': 13.67035}, 'notes': ['This polygon is an approximate circular AOI centered at the provided coordinates.', 'It was generated to have an area close to 70 hectares (target).', 'For precise analysis replace with an exact polygon/shapefile if you obtain one from local surveys or EthioTrees.']}, '/mnt/data/afedena_aoi_approx.geojson') -->