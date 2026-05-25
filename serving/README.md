# City Serving API

This folder contains a small FastAPI service that takes tag-weight inputs, embeds them through the same embedding workflow used by the city-tag scripts, and returns the nearest city ranked with the same scoring logic used in `faiss_city_global_score.py`.

## Run

```bash
pip install -r serving/requirements.txt
uvicorn serving.main:app --reload
```

The service looks for these files in `new/embeddings` by default:

- `world-subcountries-tagged.index`
- `world-subcountries-tagged-embeddings.npy`
- `world-subcountries-tagged-metadata.json`
- `embeddings_metadata.json`
- `POI.index`
- `POI_metadata.json`
- `POI_texts.json`

`world-subcountries-tagged.csv` in the repository root is only used as a supplemental source for `tourist_value_score`.

If FAISS is available, the service will use `world-subcountries-tagged.index` and `POI.index`. If not, it falls back to the NumPy city embeddings file for city ranking.

## Request

Send either a raw array or an object with a `tags` field.

Raw array example:

```json
[
  {"beach": 0.9},
  {"family-friendly": 0.7},
  {"shopping": 0.4}
]
```

Object example:

```json
{
  "tags": [
    {"tag": "beach", "weight": 0.9},
    {"tag": "family-friendly", "weight": 0.7}
  ],
  "top_k": 5,
  "candidate_k": 120,
  "description_weight": 1.0,
  "poi_weight": 1.0,
  "tags_weight": 1.0
}
```

## Endpoint

- `POST /nearest-city`
- `GET /health`
