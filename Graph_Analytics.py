"""
Graph Analytics — Winter Olympics & Snowfall
=============================================
Applies the graph analytics techniques from Lab 4 (Week 7) to the
Winter Olympics dataset.

Graph structure:
  Nodes  — countries (ISO3 code), with attributes:
             total_medals, avg_snowfall, gdp_per_capita, region
  Edges  — two countries are connected if they competed in the same
             Olympic Games AND their medal counts are within a
             similarity threshold (weighted by medal proximity)

Analyses performed (matching Lab 4 content):
  1. Graph construction + basic stats
  2. Node-level metrics: degree, betweenness, closeness, PageRank
  3. Community detection: greedy modularity (Louvain-style)
  4. Path analytics: shortest path between two countries
  5. Featurization: use centrality scores as features in logistic
     regression to predict whether a country wins any medal
     (relational logistic regression — as taught in lecture)

Outputs:
  data/figures/G1_olympic_graph.png        — graph coloured by community
  data/figures/G2_centrality_maps.png      — betweenness / PageRank maps
  data/figures/G3_degree_vs_medals.png     — degree centrality vs medals
  data/figures/G4_logistic_with_graph.png  — ROC with graph features

Usage:
  python graph_analytics.py
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, classification_report
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CLEAN_DIR = Path("data/clean")
FIG_DIR   = Path("data/figures")
MASTER    = CLEAN_DIR / "master.csv"

# Graph edge: connect two countries if their average medal difference
# across shared Games is within this threshold
MEDAL_SIMILARITY_THRESHOLD = 2

# Colour palette matching the lab notebook style
COLORS = ["#00B4D8", "#F4A261", "#06D6A0", "#7B61FF",
          "#E63946", "#FFD166", "#FF006E", "#8ECAE6"]

plt.rcParams.update({
    "figure.facecolor":  "#0D1B2A",
    "axes.facecolor":    "#1B2B3C",
    "text.color":        "white",
    "axes.labelcolor":   "white",
    "xtick.color":       "white",
    "ytick.color":       "white",
    "axes.edgecolor":    "#7FA8C0",
    "axes.grid":         True,
    "grid.color":        "#1e3a4a",
    "grid.linewidth":    0.5,
    "font.size":         11,
})

# ---------------------------------------------------------------------------
# 1. Load data and build country-level summary
# ---------------------------------------------------------------------------

def load_country_summary() -> pd.DataFrame:
    # Actual columns in master.csv:
    # country, noc_code, team_name, year, n_athletes, host_flag,
    # gold, silver, bronze, total_medals, snowfall_total,
    # snowfall_mean_gridcell, gdp, gdp_per_capita, population,
    # won_any_medal, log_total_medals, log_gdp, log_population,
    # medals_per_million
    df = pd.read_csv(MASTER)

    summary = (
        df.groupby("country")
        .agg(
            total_medals = ("total_medals",          "sum"),
            avg_snowfall = ("snowfall_mean_gridcell", "mean"),
            avg_gdp_pc   = ("gdp_per_capita",         "mean"),
            n_games      = ("year",                   "nunique"),
            team_name    = ("team_name",              "first"),
        )
        .reset_index()
        .dropna(subset=["avg_snowfall", "avg_gdp_pc"])
    )
    summary["won_any"] = (summary["total_medals"] > 0).astype(int)
    summary["log_gdp"] = np.log10(summary["avg_gdp_pc"].clip(lower=1))

    print(f"[LOAD] {len(summary)} countries with complete data")
    print(f"       {summary['won_any'].sum()} won at least one medal")
    return summary


def load_games_data() -> pd.DataFrame:
    """Return raw per-country-per-year medal data for edge construction."""
    df = pd.read_csv(MASTER)
    return df[["country", "year", "total_medals"]].dropna()

# ---------------------------------------------------------------------------
# 2. Build the graph
# ---------------------------------------------------------------------------

def build_graph(summary: pd.DataFrame, games: pd.DataFrame) -> nx.Graph:
    """
    Nodes: countries with attribute data.
    Edges: countries that competed in the same year, connected with a weight
           that reflects how similar their medal counts were that year.
           We keep only edges where medal difference <= threshold (so the
           graph is a meaningful similarity network, not a complete graph).
    """
    G = nx.Graph()

    # Add nodes with attributes
    for _, row in summary.iterrows():
        G.add_node(
            row["country"],
            total_medals = int(row["total_medals"]),
            avg_snowfall = float(row["avg_snowfall"]),
            log_gdp      = float(row["log_gdp"]),
            won_any      = int(row["won_any"]),
            name         = str(row["team_name"]),
        )

    # Add edges: per-year co-participation with medal similarity
    edge_weights = {}  # (c1, c2) -> list of similarities
    for year, group in games.groupby("year"):
        countries_this_year = group.set_index("country")["total_medals"].to_dict()
        countries = [c for c in countries_this_year if c in G.nodes]
        for i, c1 in enumerate(countries):
            for c2 in countries[i + 1:]:
                diff = abs(countries_this_year[c1] - countries_this_year[c2])
                if diff <= MEDAL_SIMILARITY_THRESHOLD:
                    key = tuple(sorted([c1, c2]))
                    edge_weights.setdefault(key, []).append(1.0 / (1.0 + diff))

    # Average similarity across shared Games → final edge weight
    for (c1, c2), weights in edge_weights.items():
        G.add_edge(c1, c2, weight=float(np.mean(weights)))

    print(f"\n[GRAPH] Nodes : {G.number_of_nodes()}")
    print(f"[GRAPH] Edges : {G.number_of_edges()}")
    print(f"[GRAPH] Is directed: {nx.is_directed(G)}")
    print(f"[GRAPH] Density: {nx.density(G):.4f}")

    if nx.is_connected(G):
        print(f"[GRAPH] Diameter: {nx.diameter(G)}")
        print(f"[GRAPH] Avg path length: {nx.average_shortest_path_length(G):.2f}")
    else:
        lcc = G.subgraph(max(nx.connected_components(G), key=len))
        print(f"[GRAPH] Graph not fully connected — largest component: "
              f"{lcc.number_of_nodes()} nodes")
        print(f"[GRAPH] LCC diameter: {nx.diameter(lcc)}")
        print(f"[GRAPH] LCC avg path length: "
              f"{nx.average_shortest_path_length(lcc):.2f}")

    return G

# ---------------------------------------------------------------------------
# 3. Node-level metrics (matching Lab 4, Section 1)
# ---------------------------------------------------------------------------

def compute_metrics(G: nx.Graph) -> dict:
    print("\n[METRICS] Computing node-level centrality metrics...")

    degrees     = dict(G.degree(weight="weight"))
    betweenness = nx.betweenness_centrality(G, weight="weight", normalized=True)
    closeness   = nx.closeness_centrality(G)
    pagerank    = nx.pagerank(G, alpha=0.85, weight="weight")
    clustering  = nx.clustering(G, weight="weight")

    metrics = {
        "degree":      degrees,
        "betweenness": betweenness,
        "closeness":   closeness,
        "pagerank":    pagerank,
        "clustering":  clustering,
    }

    # Print top 10 by degree (like lab notebook)
    print("\nTop 10 countries by degree (number of similar-medal neighbours):")
    for country, deg in sorted(degrees.items(), key=lambda x: -x[1])[:10]:
        bar = "█" * min(int(deg), 40)
        print(f"  {country}: {bar} ({deg:.1f})")

    print("\nTop 5 by Betweenness Centrality:")
    for c, s in sorted(betweenness.items(), key=lambda x: -x[1])[:5]:
        print(f"  {c}: {s:.4f}")

    print("\nTop 5 by PageRank:")
    for c, s in sorted(pagerank.items(), key=lambda x: -x[1])[:5]:
        print(f"  {c}: {s:.4f}")

    # Graph-level stats
    avg_clust = nx.average_clustering(G)
    density   = nx.density(G)
    print(f"\nAverage clustering coefficient : {avg_clust:.4f}")
    print(f"Graph density                  : {density:.4f}")

    return metrics

# ---------------------------------------------------------------------------
# 4. Community detection — greedy modularity (as taught in Week 7)
# ---------------------------------------------------------------------------

def detect_communities(G: nx.Graph) -> dict:
    """
    Uses networkx greedy modularity communities algorithm.
    This is equivalent in spirit to the Louvain method taught in lecture:
    it greedily optimises modularity by merging communities.
    """
    print("\n[COMMUNITIES] Running greedy modularity community detection...")
    communities = nx.community.greedy_modularity_communities(G, weight="weight")
    communities = [set(c) for c in communities]

    # Assign community label to each node
    community_map = {}
    for idx, comm in enumerate(communities):
        for node in comm:
            community_map[node] = idx

    print(f"  Found {len(communities)} communities")
    for i, comm in enumerate(sorted(communities, key=len, reverse=True)):
        medals = sum(G.nodes[n]["total_medals"] for n in comm)
        snowy  = np.mean([G.nodes[n]["avg_snowfall"] for n in comm])
        print(f"  Community {i+1}: {len(comm):3d} countries | "
              f"total medals: {medals:5d} | avg snowfall: {snowy:.1f} mm")

    # Compute modularity score
    modularity = nx.community.modularity(G, communities, weight="weight")
    print(f"\n  Modularity score: {modularity:.4f}  "
          f"(higher = more distinct communities)")

    return community_map

# ---------------------------------------------------------------------------
# 5. Path analytics — Dijkstra (as taught in Lab 4, Part B)
# ---------------------------------------------------------------------------

def path_analytics(G: nx.Graph) -> None:
    """
    Find shortest paths between interesting country pairs.
    Uses nx.dijkstra_path — matching the lab's Dijkstra section.
    Edge weight here represents DISSIMILARITY (1 - similarity),
    so shorter path = more similar chain of countries.
    """
    print("\n[PATHS] Shortest path analysis (Dijkstra)...")

    # Add dissimilarity as a path weight (1 - edge weight, clipped to avoid 0)
    for u, v, d in G.edges(data=True):
        G[u][v]["path_weight"] = max(1.0 - d["weight"], 0.01)

    interesting_pairs = [
        ("NOR", "IND"),  # Norway (snowy, many medals) vs India (warm, no medals)
        ("CAN", "AUS"),  # Canada vs Australia
        ("FIN", "BRA"),  # Finland vs Brazil
    ]

    for src, tgt in interesting_pairs:
        if src not in G.nodes or tgt not in G.nodes:
            print(f"  {src} or {tgt} not in graph, skipping")
            continue
        try:
            path = nx.dijkstra_path(G, src, tgt, weight="path_weight")
            length = nx.dijkstra_path_length(G, src, tgt, weight="path_weight")
            print(f"\n  {src} → {tgt}:")
            print(f"    Path   : {' → '.join(path)}")
            print(f"    Length : {length:.3f}")
        except nx.NetworkXNoPath:
            print(f"  No path between {src} and {tgt}")

# ---------------------------------------------------------------------------
# 6. Relational logistic regression (as taught in lecture)
#    — combine node attributes WITH graph features to predict won_any_medal
# ---------------------------------------------------------------------------

def relational_logistic_regression(G: nx.Graph, metrics: dict,
                                   community_map: dict) -> None:
    """
    Lecture slide: 'Relational logistic regression — combine local attributes
    with network attributes in a single logistic regression model.'

    Local attributes: avg_snowfall, log_gdp
    Network attributes: degree, betweenness, PageRank, community label
    """
    print("\n[RELATIONAL LOGISTIC] Fitting logistic regression with graph features...")

    rows = []
    for node, data in G.nodes(data=True):
        rows.append({
            "country":     node,
            "won_any":     data["won_any"],
            "avg_snowfall":data["avg_snowfall"],
            "log_gdp":     data["log_gdp"],
            "degree":      metrics["degree"][node],
            "betweenness": metrics["betweenness"][node],
            "pagerank":    metrics["pagerank"][node],
            "clustering":  metrics["clustering"][node],
            "community":   community_map.get(node, -1),
        })
    df = pd.DataFrame(rows).dropna()

    feature_cols = ["avg_snowfall", "log_gdp",
                    "degree", "betweenness", "pagerank", "clustering"]
    X = df[feature_cols].values
    y = df["won_any"].values

    if len(np.unique(y)) < 2:
        print("  [SKIP] Only one class present, cannot fit logistic regression")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=500, random_state=42)
    clf.fit(X_train, y_train)

    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)

    print("\n  Classification report:")
    print(classification_report(y_test, y_pred,
                                target_names=["no medal", "any medal"]))

    # Coefficient table
    print(f"\n  {'Feature':<20} {'Coefficient':>12}")
    print(f"  {'-'*34}")
    for name, coef in zip(feature_cols, clf.coef_[0]):
        print(f"  {name:<20} {coef:>12.4f}")

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    print(f"\n  AUC: {roc_auc:.3f}")

    # Save ROC plot
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#00B4D8", linewidth=2,
            label=f"ROC (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#7FA8C0", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Relational Logistic Regression\n(node attrs + graph features)")
    ax.legend(fontsize=10)
    fig.tight_layout()
    path = FIG_DIR / "G4_logistic_with_graph.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")

# ---------------------------------------------------------------------------
# 7. Plots
# ---------------------------------------------------------------------------

def plot_graph_communities(G: nx.Graph, community_map: dict,
                           pos: dict) -> None:
    """Visualise the graph coloured by detected community — like Lab 4 Section 1."""
    n_comm = len(set(community_map.values()))
    color_list = [COLORS[community_map.get(n, 0) % len(COLORS)] for n in G.nodes()]

    # Node size proportional to total medals
    medal_vals = [G.nodes[n]["total_medals"] for n in G.nodes()]
    max_medals = max(medal_vals) if max(medal_vals) > 0 else 1
    node_sizes = [50 + (m / max_medals) * 400 for m in medal_vals]

    fig, ax = plt.subplots(figsize=(14, 9))
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.15,
                           edge_color="#7FA8C0", width=0.5)
    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color=color_list,
                           node_size=node_sizes, alpha=0.9)

    # Label only medal-winning countries
    medal_labels = {n: n for n in G.nodes() if G.nodes[n]["total_medals"] > 5}
    nx.draw_networkx_labels(G, pos, labels=medal_labels, ax=ax,
                            font_size=6, font_color="white")

    # Legend for communities
    handles = [mpatches.Patch(color=COLORS[i % len(COLORS)],
                               label=f"Community {i+1}")
               for i in range(min(n_comm, len(COLORS)))]
    ax.legend(handles=handles, loc="lower right",
              fontsize=8, framealpha=0.3)

    ax.set_title(
        "Winter Olympics Country Graph — Communities detected by greedy modularity\n"
        "Node size = total medals  |  Edges = similar medal performance",
        fontsize=11, pad=12,
    )
    ax.axis("off")
    fig.tight_layout()
    path = FIG_DIR / "G1_olympic_graph.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")


def plot_centrality_maps(G: nx.Graph, metrics: dict, pos: dict) -> None:
    """Three-panel centrality plot — mirrors Lab 4 Cell 12."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    titles  = ["Betweenness Centrality", "Closeness Centrality", "PageRank"]
    keys    = ["betweenness", "closeness", "pagerank"]
    cmaps   = ["Purples", "Blues", "Oranges"]

    for ax, title, key, cmap in zip(axes, titles, keys, cmaps):
        values = [metrics[key][n] for n in G.nodes()]
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.1,
                               edge_color="#7FA8C0", width=0.4)
        sc = nx.draw_networkx_nodes(G, pos, ax=ax,
                                    node_color=values, cmap=cmap,
                                    node_size=60, alpha=0.9)
        # Label top 5 nodes
        top5 = sorted(metrics[key], key=metrics[key].get, reverse=True)[:5]
        nx.draw_networkx_labels(G, pos,
                                labels={n: n for n in top5},
                                ax=ax, font_size=7, font_color="white")
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    fig.suptitle("Node Centrality Maps — Winter Olympics Country Graph",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = FIG_DIR / "G2_centrality_maps.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")


def plot_degree_vs_medals(G: nx.Graph, metrics: dict) -> None:
    """Scatter: degree centrality vs total medals — mirrors Lab 4 Cell 15."""
    degrees = metrics["degree"]
    nodes   = list(G.nodes())
    x = [degrees[n] for n in nodes]
    y = [G.nodes[n]["total_medals"] for n in nodes]

    # Colour by won_any
    colors = ["#F4A261" if G.nodes[n]["won_any"] else "#7FA8C0" for n in nodes]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(x, y, c=colors, s=60, alpha=0.8,
               edgecolors="white", linewidths=0.4)

    # Annotate top 10 by medals
    top10 = sorted(nodes, key=lambda n: G.nodes[n]["total_medals"], reverse=True)[:10]
    for n in top10:
        ax.annotate(f"  {n}", (degrees[n], G.nodes[n]["total_medals"]),
                    fontsize=7, color="#FFD166", fontweight="bold")

    # Correlation line
    m, b = np.polyfit(x, y, 1)
    x_line = np.linspace(min(x), max(x), 100)
    ax.plot(x_line, m * x_line + b, color="#E63946",
            linewidth=1.5, linestyle="--", label=f"Trend (slope={m:.1f})")

    ax.set_xlabel("Weighted Degree (similar-medal neighbours)")
    ax.set_ylabel("Total Olympic Medals (all years)")
    ax.set_title("Degree Centrality vs Total Medals\n"
                 "Orange = won at least one medal")
    ax.legend(fontsize=9)
    legend_handles = [
        mpatches.Patch(color="#F4A261", label="Won medals"),
        mpatches.Patch(color="#7FA8C0", label="No medals"),
    ]
    ax.legend(handles=legend_handles, fontsize=9)

    fig.tight_layout()
    path = FIG_DIR / "G3_degree_vs_medals.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────
    summary = load_country_summary()
    games   = load_games_data()

    # ── Build graph ───────────────────────────────────────────────────────
    G = build_graph(summary, games)

    # Layout computed once, reused in all plots
    pos = nx.spring_layout(G, seed=42, k=0.4)

    # ── Metrics ───────────────────────────────────────────────────────────
    metrics = compute_metrics(G)

    # ── Community detection ───────────────────────────────────────────────
    community_map = detect_communities(G)

    # ── Path analytics ────────────────────────────────────────────────────
    path_analytics(G)

    # ── Relational logistic regression ────────────────────────────────────
    relational_logistic_regression(G, metrics, community_map)

    # ── Plots ─────────────────────────────────────────────────────────────
    print("\n[PLOTS]")
    plot_graph_communities(G, community_map, pos)
    plot_centrality_maps(G, metrics, pos)
    plot_degree_vs_medals(G, metrics)

    print(f"\n✓ Graph analytics complete → {FIG_DIR}/")
    print("  G1_olympic_graph.png      — community structure")
    print("  G2_centrality_maps.png    — betweenness / closeness / PageRank")
    print("  G3_degree_vs_medals.png   — degree centrality vs medals")
    print("  G4_logistic_with_graph.png — ROC for relational logistic regression")


if __name__ == "__main__":
    main()