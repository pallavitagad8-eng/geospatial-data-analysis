"""
Geospatial Data Analysis - Workora Internship (synthetic data)
Objective: Visualize sales/market data geographically and detect underserved regional markets.
Run:  python geospatial_analysis.py
Needs: pandas, matplotlib, reportlab   (optional: folium for the Folium map)
"""
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

RAW = "regional_market.csv"

# ---------- STEP 1: CLEAN ----------
df = pd.read_csv(RAW)
log = []
n0 = len(df)
for c in ["RegionID", "City", "State", "PopulationSegment"]:
    df[c] = df[c].astype(str).str.strip()
log.append(f"Rows loaded: {n0}")
log.append(f"Missing values: {int(df.isna().sum().sum())}")
dups = int(df.duplicated().sum()); df = df.drop_duplicates()
log.append(f"Duplicate rows removed: {dups}")
dup_id = int(df.duplicated('RegionID').sum()); df = df.drop_duplicates('RegionID')
log.append(f"Duplicate RegionID removed: {dup_id}")
bad_geo = ~df.Latitude.between(6, 37) | ~df.Longitude.between(68, 98)
log.append(f"Coordinates outside India bounding box: {int(bad_geo.sum())}")
df = df[~bad_geo]
bad_rate = ~df.LeadConversionRate.between(0, 1)
log.append(f"Invalid conversion rates: {int(bad_rate.sum())}")
df = df[~bad_rate]
log.append(f"City-State mapping consistent: {df.groupby('City').State.nunique().max() == 1}")
log.append(f"Rows after cleaning: {len(df)}")
df.to_csv("regional_market_cleaned.csv", index=False)

# ---------- STEP 2: AGGREGATE ----------
def mm(s): return (s - s.min()) / (s.max() - s.min())
df["DemandPerCompetitor"] = df.MonthlySearchDemand / df.ExistingCompetitors
df["OpportunityScore"] = 100 * (
    0.30 * mm(df.MonthlySearchDemand) + 0.20 * mm(df.LeadConversionRate) +
    0.15 * mm(df.MedianAnnualIncomeINR) + 0.20 * (1 - mm(df.ExistingCompetitors)) +
    0.15 * (1 - mm(df.EstimatedAcquisitionCostINR)))
df["PotentialLeads"] = df.MonthlySearchDemand * df.LeadConversionRate
city = df.groupby(["City", "State"]).agg(
    Regions=("RegionID", "count"), TotalDemand=("MonthlySearchDemand", "sum"),
    AvgCompetitors=("ExistingCompetitors", "mean"), AvgIncomeINR=("MedianAnnualIncomeINR", "mean"),
    AvgConversion=("LeadConversionRate", "mean"), AvgCAC_INR=("EstimatedAcquisitionCostINR", "mean"),
    PotentialLeads=("PotentialLeads", "sum"), OpportunityScore=("OpportunityScore", "mean"),
    Lat=("Latitude", "mean"), Lon=("Longitude", "mean")).reset_index()
city["DemandPerCompetitor"] = city.TotalDemand / (city.AvgCompetitors * city.Regions)
city = city.sort_values("OpportunityScore", ascending=False).reset_index(drop=True)
city.round(3).to_csv("city_summary.csv", index=False)
df.round(3).to_csv("regional_market_scored.csv", index=False)
top3 = city.head(3)
seg = df.groupby("PopulationSegment").agg(Regions=("RegionID","count"), Score=("OpportunityScore","mean"),
                                          Demand=("MonthlySearchDemand","mean")).round(2).reset_index()
top_regions = df.sort_values("OpportunityScore", ascending=False).head(10)

# ---------- STEP 3: VISUALS ----------
plt.figure(figsize=(7, 6))
sc = plt.scatter(df.Longitude, df.Latitude, c=df.OpportunityScore, s=df.MonthlySearchDemand / 15,
                 cmap="RdYlGn", alpha=.75, edgecolor="k", linewidth=.3)
plt.colorbar(sc, label="Opportunity Score")
for _, r in city.iterrows(): plt.text(r.Lon + .25, r.Lat + .25, r.City, fontsize=8)
plt.xlabel("Longitude"); plt.ylabel("Latitude"); plt.title("Spatial scatter: colour = score, size = search demand")
plt.tight_layout(); plt.savefig("fig_map.png", dpi=150); plt.close()

plt.figure(figsize=(7, 4))
c2 = city.sort_values("OpportunityScore")
plt.barh(c2.City, c2.OpportunityScore, color=["#2e7d32" if x in set(top3.City) else "#90a4ae" for x in c2.City])
plt.xlabel("Avg Opportunity Score"); plt.title("City ranking (top 3 highlighted)")
plt.tight_layout(); plt.savefig("fig_rank.png", dpi=150); plt.close()

plt.figure(figsize=(7, 4))
plt.scatter(city.AvgCompetitors, city.TotalDemand / city.Regions, s=city.Regions * 8, alpha=.7)
for _, r in city.iterrows(): plt.text(r.AvgCompetitors + .03, r.TotalDemand / r.Regions, r.City, fontsize=8)
plt.xlabel("Avg competitors per region"); plt.ylabel("Avg monthly search demand per region")
plt.title("Demand vs competition by city"); plt.tight_layout(); plt.savefig("fig_gap.png", dpi=150); plt.close()

# ---------- STEP 4: INTERACTIVE HTML MAP (Leaflet) ----------
pts = df[["RegionID","City","State","Latitude","Longitude","PopulationSegment","MonthlySearchDemand",
          "ExistingCompetitors","LeadConversionRate","EstimatedAcquisitionCostINR","OpportunityScore"]].round(3).to_dict("records")
top3_names = list(top3.City)
html = """<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Geospatial Analysis - Regional Market (Synthetic Data)</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>body{margin:0;font-family:sans-serif}#map{height:100vh}.box{position:absolute;z-index:999;top:10px;right:10px;background:#fff;padding:10px;border-radius:8px;font-size:13px;max-width:240px;box-shadow:0 1px 6px #0004}</style></head>
<body><div id="map"></div><div class="box"><b>Top 3 high-potential cities</b><br>__TOP__<br><br>Circle colour = Opportunity Score (green high), size = search demand.<br><i>Synthetic data, for learning only.</i></div>
<script>const data=__DATA__;
const map=L.map('map').setView([21,79],5);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{attribution:'&copy; OpenStreetMap'}).addTo(map);
function col(s){return s>60?'#1a9850':s>50?'#a6d96a':s>40?'#fee08b':s>30?'#fdae61':'#d73027'}
data.forEach(d=>{L.circleMarker([d.Latitude,d.Longitude],{radius:3+d.MonthlySearchDemand/250,color:'#333',weight:.5,fillColor:col(d.OpportunityScore),fillOpacity:.75})
.bindPopup(`<b>${d.RegionID} - ${d.City}, ${d.State}</b><br>Segment: ${d.PopulationSegment}<br>Demand: ${d.MonthlySearchDemand}<br>Competitors: ${d.ExistingCompetitors}<br>Conversion: ${(d.LeadConversionRate*100).toFixed(1)}%<br>CAC: Rs ${d.EstimatedAcquisitionCostINR}<br>Score: ${d.OpportunityScore}`).addTo(map)});
</script></body></html>"""
html = html.replace("__DATA__", json.dumps(pts)).replace("__TOP__", "<br>".join(f"{i+1}. {n}" for i, n in enumerate(top3_names)))
open("interactive_map.html", "w", encoding="utf-8").write(html)

# optional Folium version
try:
    import folium
    m = folium.Map(location=[21, 79], zoom_start=5)
    for d in pts:
        s = d["OpportunityScore"]
        folium.CircleMarker([d["Latitude"], d["Longitude"]], radius=3 + d["MonthlySearchDemand"] / 250,
            color="#333", weight=.5, fill=True, fill_opacity=.75,
            fill_color="#1a9850" if s > 60 else "#a6d96a" if s > 50 else "#fee08b" if s > 40 else "#fdae61" if s > 30 else "#d73027",
            popup=f'{d["RegionID"]} {d["City"]} score={s}').add_to(m)
    m.save("folium_map.html")
except ImportError:
    print("folium not installed - skipped folium_map.html (pip install folium)")

# ---------- STEP 5: PDF REPORT ----------
ss = getSampleStyleSheet()
doc = SimpleDocTemplate("Spatial_Analysis_Report.pdf", pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
S = []
P = lambda t, s="BodyText": S.append(Paragraph(t, ss[s]))
def tbl(data, widths=None):
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#37474f")),("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("GRID",(0,0),(-1,-1),.4,colors.grey),("FONTSIZE",(0,0),(-1,-1),8),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#eceff1")])]))
    S.append(t); S.append(Spacer(1, 8))
P("Geospatial Data Analysis - Spatial Analysis Report", "Title")
P("Objective: visualize sales/market data geographically and detect underserved regional markets.")
P("<i>Dataset: regional_market.csv (synthetic, for learning only - not real Workora business results).</i>")
P("1. Data Cleaning", "Heading2")
for l in log: P("&bull; " + l)
P("Result: the file was already clean (no missing values, duplicates or invalid coordinates). "
  "Text fields were trimmed and checked; the cleaned file is saved as regional_market_cleaned.csv. "
  "Location identifiers used: State, City, Latitude/Longitude (no postal code column exists in this dataset).")
P("2. Aggregation by Geographic Region", "Heading2")
P("Revenue is not present in this dataset, so market potential is measured by search demand, expected leads "
  "(demand x lead conversion rate), income, competition and acquisition cost. Calculated fields: "
  "<b>DemandPerCompetitor</b>, <b>PotentialLeads</b> and <b>OpportunityScore</b> = 100 x (0.30 demand + 0.20 conversion + "
  "0.15 income + 0.20 (1 - competitors) + 0.15 (1 - acquisition cost)), each min-max normalised. Weights are an assumption.")
rows = [["City","Regions","Total demand","Avg comp.","Avg conv.","Avg CAC (Rs)","Score"]]
for _, r in city.iterrows():
    rows.append([r.City, r.Regions, int(r.TotalDemand), f"{r.AvgCompetitors:.1f}", f"{r.AvgConversion*100:.1f}%", f"{r.AvgCAC_INR:.0f}", f"{r.OpportunityScore:.1f}"])
tbl(rows)
P("3. Spatial Visualization", "Heading2")
S.append(Image("fig_map.png", width=14*cm, height=12*cm)); S.append(PageBreak())
S.append(Image("fig_rank.png", width=14*cm, height=8*cm)); S.append(Image("fig_gap.png", width=14*cm, height=8*cm))
P("An interactive map is provided in interactive_map.html.")
P("4. Top 3 High-Potential Underserved Regions (cities)", "Heading2")
rows = [["Rank","City","State","Score","Total demand","Avg comp.","Avg conv.","Avg CAC"]]
for i, r in top3.iterrows():
    rows.append([i+1, r.City, r.State, f"{r.OpportunityScore:.1f}", int(r.TotalDemand), f"{r.AvgCompetitors:.1f}", f"{r.AvgConversion*100:.1f}%", f"Rs {r.AvgCAC_INR:.0f}"])
tbl(rows)
P("Best individual micro-regions (top 10 by score):")
rows = [["RegionID","City","Segment","Demand","Comp.","Conv.","CAC","Score"]]
for _, r in top_regions.iterrows():
    rows.append([r.RegionID, r.City, r.PopulationSegment, r.MonthlySearchDemand, r.ExistingCompetitors, f"{r.LeadConversionRate*100:.1f}%", r.EstimatedAcquisitionCostINR, f"{r.OpportunityScore:.1f}"])
tbl(rows)
P("5. Conclusion & Assumptions", "Heading2")
P(f"The three cities with the highest average opportunity score are <b>{', '.join(top3_names)}</b>. "
  "Scores depend on the chosen weights; the data is synthetic, so results are for learning only.")
doc.build(S)
print("\n".join(log)); print(city[["City","OpportunityScore"]].round(1).to_string()); print(top3_names)
