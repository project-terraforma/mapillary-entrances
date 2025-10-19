# mapillary-entrances

Julien Howard and Evan Rantala

This project integrates Overture Buildings (footprints) and Overture Places (points of interest) with Mapillary street-level imagery to detect and map building entrances, enabling more accurate real-world access mapping.



```bash
git clone <repo-url>
cd mapillary-entrances
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Mapillary
export MAPILLARY_ACCESS_TOKEN=YOUR_TOKEN_HERE
# (optional) install Overture CLI for local extracts:
# pipx install overturemaps

# Run (SoMa sample)
PYTHONUNBUFFERED=1 PYTHONPATH=. python3 -m src.cli_plumbing \
  --bbox="-122.4045,37.7825,-122.3965,37.7895" \
  --radius-m=120 \
  --place-radius-m=60 \
  --limit-buildings=50 \
  --max-images-per-building=8 \
  --prefer-360 \
  --src-mode=auto