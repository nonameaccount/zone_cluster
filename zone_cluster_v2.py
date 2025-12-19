#!/usr/bin/env python3
"""
Geo Zone Clusterer (Configurable)
---------------------------------
- Geocodes with: OpenCage, Geoapify, Google Maps. ("qgis" option is a stub)
- Clusters addresses into K zones (K auto-chosen in [--kmin, --kmax] via silhouette score)
- Outputs: CSV, Excel (one sheet per zone + Summary), Folium (Leaflet) map
- Optional: Google Maps JS page you can host and share as a URL

Examples
--------
OpenCage:
  python zone_cluster_v2.py --input addrs.csv --city "Atlanta, GA" --kmin 6 --kmax 8 \
    --geocoder opencage --opencage-key YOUR_OPENCAGE_KEY

Geoapify:
  python zone_cluster_v2.py --input addrs.csv --city "Dallas, TX" --kmin 6 --kmax 8 \
    --geocoder geoapify --geoapify-key YOUR_GEOAPIFY_KEY

Google + generate a Google Maps HTML:
  python zone_cluster_v2.py --input addrs.csv --city "New York, NY" --kmin 6 --kmax 8 \
    --geocoder google --google-key YOUR_GOOGLE_KEY --make-google-map

Notes
-----
- If your CSV already has lat,lon columns, geocoding is skipped.
- QGIS plugins aren’t web APIs; if you choose `qgis` the script exits with instructions.
"""
import argparse, os, sys, math, json, time, urllib.parse, urllib.request
from typing import Optional, Tuple
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import folium
from shapely.geometry import Point, MultiPoint

# ----------------------------- clustering helpers -----------------------------

def choose_best_k(X, kmin=6, kmax=8, seed=42) -> int:
    best_k, best_score = kmin, -1
    for k in range(kmin, kmax + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=seed)
        labels = km.fit_predict(X)
        if len(set(labels)) == 1:
            continue
        score = silhouette_score(X, labels)
        if score > best_score:
            best_k, best_score = k, score
    return best_k

# ----------------------------- geocoding helpers ------------------------------

def assemble_query(row, city_context: Optional[str]) -> str:
    # First check if there's a full "Address" column (capital A)
    if "Address" in row and str(row.get("Address", "")).strip():
        q = str(row.get("Address")).strip()
        if city_context and city_context.lower() not in q.lower():
            q = f"{q}, {city_context}"
        return q

    # Otherwise, assemble from individual components
    parts = [str(row.get(c, "")).strip() for c in ["address","city","state","zip"] if str(row.get(c, "")).strip()]
    q = ", ".join(parts)
    if city_context and city_context.lower() not in q.lower():
        q = f"{q}, {city_context}"
    return q

def _http_json(url: str) -> dict:
    with urllib.request.urlopen(url) as r:
        return json.loads(r.read().decode("utf-8"))

def geocode_opencage(queries, api_key, delay=1.1):
    if not api_key:
        raise SystemExit("OpenCage selected but no --opencage-key provided.")
    out = []
    for q in queries:
        url = "https://api.opencagedata.com/geocode/v1/json?" + urllib.parse.urlencode({"q": q, "key": api_key, "limit": 1})
        data = _http_json(url)
        if data.get("results"):
            g = data["results"][0]["geometry"]
            out.append((g["lat"], g["lng"]))
        else:
            out.append((float("nan"), float("nan")))
        time.sleep(delay)
    return out

def geocode_geoapify(queries, api_key, delay=1.1):
    if not api_key:
        raise SystemExit("Geoapify selected but no --geoapify-key provided.")
    out = []
    for q in queries:
        url = "https://api.geoapify.com/v1/geocode/search?" + urllib.parse.urlencode({"text": q, "apiKey": api_key, "limit": 1})
        data = _http_json(url)
        feats = data.get("features", [])
        if feats:
            lon, lat = feats[0]["geometry"]["coordinates"]
            out.append((lat, lon))
        else:
            out.append((float("nan"), float("nan")))
        time.sleep(delay)
    return out

def geocode_google(queries, api_key, delay=0.2):
    if not api_key:
        raise SystemExit("Google selected but no --google-key provided.")
    out = []
    for q in queries:
        url = "https://maps.googleapis.com/maps/api/geocode/json?" + urllib.parse.urlencode({"address": q, "key": api_key})
        data = _http_json(url)
        res = data.get("results", [])
        if res:
            loc = res[0]["geometry"]["location"]
            out.append((loc["lat"], loc["lng"]))
        else:
            out.append((float("nan"), float("nan")))
        time.sleep(delay)
    return out

def geocode_df(df: pd.DataFrame, geocoder_name: str, keys: dict, city_context: Optional[str]) -> pd.DataFrame:
    if {"lat","lon"}.issubset(df.columns):
        return df.copy()

    queries = [assemble_query(row, city_context) for _, row in df.iterrows()]
    if geocoder_name == "opencage":
        coords = geocode_opencage(queries, keys.get("opencage"))
    elif geocoder_name == "geoapify":
        coords = geocode_geoapify(queries, keys.get("geoapify"))
    elif geocoder_name == "google":
        coords = geocode_google(queries, keys.get("google"))
    elif geocoder_name == "qgis":
        raise SystemExit(
            "QGIS geocoding is a desktop plugin, not a hosted API. "
            "Geocode in QGIS and export lat/lon, then re-run this tool (it will skip geocoding)."
        )
    else:
        raise SystemExit("Unknown geocoder: " + geocoder_name)

    lats, lons = zip(*coords) if coords else ([], [])
    out = df.copy()
    out["lat"], out["lon"] = lats, lons
    out = out.dropna(subset=["lat","lon"]).reset_index(drop=True)
    return out

# ----------------------------- map rendering ----------------------------------

def color_cycle(n):
    base = [
        "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
        "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"
    ]
    return (base * ((n // len(base)) + 1))[:n]

def make_folium_map(df: pd.DataFrame, centers: pd.DataFrame, center: Tuple[float,float], out_html: str):
    m = folium.Map(location=center, zoom_start=9, tiles="CartoDB positron", control_scale=True)
    colors = {int(z): c for z, c in zip(sorted(df["zone"].unique()), color_cycle(df["zone"].nunique()))}
    for _, r in df.iterrows():
        folium.CircleMarker((r.lat, r.lon), radius=6, color=colors[int(r.zone)],
                            fill=True, fill_opacity=0.85,
                            tooltip=f"Zone {int(r.zone)} - {r.get('name','')}").add_to(m)
    for z, sub in df.groupby("zone"):
        pts = [Point(yx[0], yx[1]) for yx in sub[["lat","lon"]].to_numpy()]
        hull = MultiPoint(pts).convex_hull
        folium.GeoJson(
            hull.__geo_interface__,
            style_function=lambda x, col=colors[int(z)]: {"color": col, "fillColor": col, "fillOpacity": 0.10, "weight": 2},
            name=f"Zone {int(z)}"
        ).add_to(m)
    for _, r in centers.iterrows():
        folium.Marker((r.lat, r.lon), tooltip=f"Zone {int(r.zone)} center").add_to(m)
    folium.LayerControl().add_to(m)
    m.save(out_html)
    return out_html

def make_google_maps_html(df: pd.DataFrame, centers: pd.DataFrame, api_key: str, out_html: str):
    if not api_key:
        raise SystemExit("Missing --google-key required for --make-google-map.")
    colors = {int(z): c for z, c in zip(sorted(df["zone"].unique()), color_cycle(df["zone"].nunique()))}
    zone_polys = {}
    for z, sub in df.groupby("zone"):
        pts = [Point(yx[1], yx[0]) for yx in sub[["lon","lat"]].to_numpy()]  # x=lon, y=lat
        hull = MultiPoint(pts).convex_hull
        if hull.geom_type == "Polygon":
            zone_polys[int(z)] = list(hull.exterior.coords)  # (lon,lat)
    pts_js = [{"name": str(r.get("name","")), "lat": float(r["lat"]), "lon": float(r["lon"]), "zone": int(r["zone"])} for _, r in df.iterrows()]
    centers_js = [{"lat": float(r["lat"]), "lon": float(r["lon"]), "zone": int(r["zone"])} for _, r in centers.iterrows()]
    hulls_js = {int(z): [{"lat": float(lat), "lng": float(lon)} for (lon,lat) in coords] for z, coords in zone_polys.items()}

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>Google Map - Cluster Zones</title>
<style>html,body,#map{{height:100%;margin:0}} .legend{{position:absolute;background:#fff;padding:8px;border:1px solid #ddd;right:12px;top:12px;font:12px/1.3 sans-serif}}</style>
<script src="https://maps.googleapis.com/maps/api/js?key={api_key}"></script>
</head>
<body>
<div id="map"></div>
<div class="legend" id="legend"></div>
<script>
const dataPoints = {json.dumps(pts_js)};
const centers = {json.dumps(centers_js)};
const hulls = {json.dumps(hulls_js)};
const zoneColors = {json.dumps(colors)};

function init() {{
  const center = {{
    lat: dataPoints.reduce((a,b)=>a+b.lat,0)/dataPoints.length,
    lng: dataPoints.reduce((a,b)=>a+b.lon,0)/dataPoints.length
  }};
  const map = new google.maps.Map(document.getElementById('map'), {{center, zoom: 10}});
  for (const p of dataPoints) {{
    new google.maps.Marker({{
      position: {{lat: p.lat, lng: p.lon}},
      map,
      title: `Zone ${'{'}p.zone{'}'} - ${'{'}p.name{'}'}`,
      icon: {{
        path: google.maps.SymbolPath.CIRCLE,
        fillColor: zoneColors[p.zone], fillOpacity: 1, strokeWeight: 1, scale: 6, strokeColor: '#333'
      }}
    }});
  }}
  for (const [zone, path] of Object.entries(hulls)) {{
    new google.maps.Polygon({{
      paths: path,
      strokeColor: zoneColors[zone], strokeOpacity: 0.9, strokeWeight: 2,
      fillColor: zoneColors[zone], fillOpacity: 0.10,
      map
    }});
  }}
  for (const c of centers) {{
    new google.maps.Marker({{
      position: {{lat: c.lat, lng: c.lon}},
      map,
      icon: 'https://maps.google.com/mapfiles/ms/icons/blue-dot.png',
      title: `Zone ${'{'}c.zone{'}'} center`
    }});
  }}
  const legend = document.getElementById('legend');
  legend.innerHTML = Object.entries(zoneColors)
    .map(([z,col]) => `<div><span style="display:inline-block;width:12px;height:12px;background:${'{'}col{'}'};margin-right:6px;border:1px solid #999"></span>Zone ${'{'}z{'}'}</div>`)
    .join('');
  map.controls[google.maps.ControlPosition.RIGHT_TOP].push(legend);
}}
window.onload = init;
</script>
</body>
</html>"""
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    return out_html

# ----------------------------- exports ----------------------------------------

def export_excel(df: pd.DataFrame, out_prefix: str):
    import xlsxwriter
    xlsx = f"{out_prefix}_zones.xlsx"
    with pd.ExcelWriter(xlsx, engine="xlsxwriter") as xw:
        for z, sub in df.groupby("zone"):
            cols = [c for c in ["zone","name","address","city","state","zip","lat","lon"] if c in sub.columns]
            sub[cols].to_excel(xw, sheet_name=f"Zone_{int(z)}", index=False)
        summary = df.groupby("zone").size().rename("count").reset_index()
        summary.to_excel(xw, sheet_name="Summary", index=False)
    return xlsx

# ----------------------------- main -------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Cluster addresses into geographic zones (configurable).")
    ap.add_argument("--input", required=True, help="CSV with name,address,city,state,zip (or lat,lon).")
    ap.add_argument("--city", default="", help="Optional context to append (e.g., 'Atlanta, GA', 'Missouri').")
    ap.add_argument("--kmin", type=int, default=6)
    ap.add_argument("--kmax", type=int, default=8)
    ap.add_argument("--geocoder", choices=["opencage","geoapify","google","qgis"], default="opencage")
    ap.add_argument("--opencage-key", default=os.getenv("OPENCAGE_KEY"))
    ap.add_argument("--geoapify-key", default=os.getenv("GEOAPIFY_KEY"))
    ap.add_argument("--google-key", default=os.getenv("GOOGLE_MAPS_KEY"))
    ap.add_argument("--out-prefix", default="zones")
    ap.add_argument("--make-google-map", action="store_true", help="Also emit a Google Maps JS HTML (requires --google-key).")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    # Handle both "Name" and "name" columns
    if "Name" in df.columns and "name" not in df.columns:
        df["name"] = df["Name"]
    elif "name" not in df.columns:
        df["name"] = None

    # Geocode
    keys = {"opencage": args.opencage_key, "geoapify": args.geoapify_key, "google": args.google_key}
    df = geocode_df(df, args.geocoder, keys, args.city)
    if df.empty:
        raise SystemExit("No rows with valid lat/lon. Check input and API keys.")

    # Cluster
    X = df[["lat","lon"]].to_numpy()
    k = choose_best_k(X, args.kmin, args.kmax, seed=42)
    km = KMeans(n_clusters=k, n_init=25, random_state=42)
    df["zone"] = km.fit_predict(X) + 1

    centers = pd.DataFrame(km.cluster_centers_, columns=["lat","lon"])
    centers["zone"] = range(1, k+1)

    # Outputs
    out_prefix = args.out_prefix
    df.sort_values(["zone","name"]).to_csv(f"{out_prefix}_zones.csv", index=False)
    xlsx = export_excel(df, out_prefix)

    center = (float(df["lat"].mean()), float(df["lon"].mean()))
    leaflet = f"{out_prefix}_leaflet_map.html"
    make_folium_map(df, centers, center, leaflet)
    # comment

    gmap = None
    if args.make_google_map:
        if not args.google_key:
            raise SystemExit("Provide --google-key for --make-google-map.")
        gmap = f"{out_prefix}_google_map.html"
        make_google_maps_html(df, centers, args.google_key, gmap)

    print(f"Done. K={k}\n- {out_prefix}_zones.csv\n- {out_prefix}_zones.xlsx\n- {leaflet}")
    if gmap:
        print(f"- {gmap} (host this file to share via URL)")

if __name__ == "__main__":
    main()
