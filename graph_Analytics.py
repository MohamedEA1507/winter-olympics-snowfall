"""
Research question
-----------------
To what extent does a country's snowy climate relate to Winter Olympic medal performance, and does that relationship remain after controlling for
GDP and population?

Role of this file
-----------------
This is a SUPPLEMENTARY exploratory analysis, not the primary hypothesis test. The main models (OLS, Poisson, Negative Binomial, Logistic Regression)
live in the main modelling script.  Here we ask a complementary graph question:
    Do countries with similar climate and economic profiles cluster together in a network,
    and do medal-winning countries tend to form tight communities within that structure?

Graph structure
---------------
  Nodes  — countries (noc_codes), with attributes:
             total_medals, avg_snowfall, log_gdp, log_population, won_any
  Edges  — two countries are connected if they are among each other's K_NEIGHBOURS nearest neighbours in the (snowfall, log_gdp,
             log_population) feature space, STANDARDISED before distance computation so no single feature dominates.
             Edge weight = 1 / (1 + Euclidean distance), so closer countries get a higher weight.

Why this edge definition?
  Building edges from climate and economic similarity means the graph structure
  is derived entirely from the EXPLANATORY variables in our research question.
  Medal outcomes are then used only as node attributes and as the prediction
  target — not to define who is connected to whom.  This avoids circularity.

Analyses performed:
------------------------------------------------------
  1. Graph construction + basic stats
  2. Node-level metrics: degree, betweenness, closeness, PageRank, clustering
  3. Community detection: greedy modularity
     Note: greedy modularity and Louvain both optimise modularity but are different algorithms.
     NetworkX ships greedy modularity; Louvain requires the python-louvain package.
  4. Path analytics: Dijkstra shortest path between interesting country pairs

Outputs (saved to data/figures/)
---------------------------------
  G1_olympic_graph.png       — graph coloured by community
  G2_centrality_maps.png     — betweenness / closeness / PageRank
  G3_degree_vs_medals.png    — degree centrality vs total medals
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# =============================================================================
# Config
# =============================================================================

CLEAN_DIR = Path("data/clean")
FIG_DIR = Path("data/figures/graph_analytics")
MASTER = CLEAN_DIR / "master.csv"

# Each country is connected to its 5 nearest neighbours in climate/economic space.
# This creates a local similarity network: countries are linked to a few close peers, not to every broadly similar country.
# This helps reveal clusters while keeping the graph connected enough for community detection and Dijkstra path analysis.
K_NEIGHBOURS = 5

# Eight visually distinct colours for community colouring. Chosen to be readable on the dark background defined in plt.rcParams below.
COLORS = ["#00B4D8", "#F4A261", "#06D6A0", "#7B61FF", "#E63946", "#FFD166", "#FF006E", "#8ECAE6"]

# Dark-theme matplotlib style — consistent with the lab notebook palette.
plt.rcParams.update({
    "figure.facecolor": "#0D1B2A",  # dark navy background for figures
    "axes.facecolor": "#1B2B3C",  # slightly lighter for axes area
    "text.color": "white",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "axes.edgecolor": "#7FA8C0",  # muted blue-grey for axis borders
    "axes.grid": True,
    "grid.color": "#1e3a4a",  # subtle grid lines
    "grid.linewidth": 0.5,
    "font.size": 11,
})

# =============================================================================
# 1. Load data — aggregate master.csv to one row per country
# =============================================================================
def load_country_summary() -> pd.DataFrame:
    """
    Aggregate master.csv to one row per country.

    Node IDs are noc_code (the 3-letter Olympic/NOC code, e.g. NOR, IND, CAN).
    This is what the Dijkstra pairs reference. country_name is kept as a readable label for print output.

    We prefer noc_code over country (full name) because the Dijkstra path pairs use 3-letter codes.
    If noc_code is absent the script falls back to the country column and prints a warning.
    """
    df = pd.read_csv(MASTER)
    id_col = "noc_code" if "noc_code" in df.columns else "country"

    summary = (
        df.groupby(id_col)
        .agg(
            total_medals=("total_medals", "sum"), # Sum medals across all Games
            avg_snowfall=("snowfall_mean_gridcell", "mean"),  # Average snowfall and climate variables across years, since climate is broadly stable over the period of study.
            avg_gdp=("gdp", "mean"), # Average total GDP across years a country participated. We average (rather than sum) because total GDP is a stock variable — it does not accumulate across Games like medals do.
            avg_population=("population", "mean"),  # Average population across years
            n_games=("year", "nunique"), # Count of distinct Games the country participated in
            country_name=("country", "first"),
            team_name=("team_name", "first"),
        )
        .reset_index()
        .rename(columns={id_col: "node_id"}) # Rename so the rest of the code always refers to node_id regardless of whether it came from noc_code or country.
        .dropna(subset=["avg_snowfall", "avg_gdp", "avg_population"]) # Drop rows missing any of the three features used to build the graph. A country without snowfall, GDP, or population data cannot be placed meaningfully in the similarity space.
    )

    summary["won_any"] = (summary["total_medals"] > 0).astype(int) # Binary medal flag used as the classification target later.

    # Log10 of total GDP and population — see docstring for rationale. clip(lower=1) prevents log(0) for countries with near-zero values.
    summary["log_gdp"] = np.log10(summary["avg_gdp"].clip(lower=1))
    summary["log_population"] = np.log10(summary["avg_population"].clip(lower=1))

    print(f"[LOAD] {len(summary)} countries with complete data  " f"(ID column: '{id_col}')")
    print(f"       {summary['won_any'].sum()} won at least one medal")
    return summary


# ---------------------------------------------------------------------------
# 2. Build the graph — climate / economic similarity network
# ---------------------------------------------------------------------------
def build_graph(summary: pd.DataFrame) -> nx.Graph:
    """
    Build an undirected weighted graph where every node is a country and every edge represents climate/economic similarity — NOT medal similarity.

    Algorithm
    ---------
    1. Extract a feature vector per country: [avg_snowfall,  log_gdp,  log_population]
    2. Standardise to zero mean / unit variance so that GDP (millions of USD) and snowfall (mm) contribute equally to the distance metric.
    3. Fit a k-nearest-neighbour index and retrieve the K_NEIGHBOURS closest countries for each country in the standardised space.
    4. For each (country, neighbour) pair add one undirected edge storing:
           distance = Euclidean distance in standardised space  (used by shortest-path metrics: smaller = more similar)
           weight   = 1 / (1 + distance)  (used by flow metrics: larger = more similar)

    Why k-NN instead of a threshold?
      A fixed distance threshold would produce very different numbers of edges for dense vs sparse regions of the feature space.
      k-NN guarantees every country has at least K_NEIGHBOURS connections regardless of where it sits, making degree and centrality metrics more comparable across countries.
    """
    G = nx.Graph()

    # ── Add nodes ──────────────────────────────────────────────────────────
    # Each node stores the attributes we will later use for colouring, sizing, and as features in the logistic regression.
    for _, row in summary.iterrows():
        G.add_node(
            row["node_id"],  # e.g. "NOR", "IND", "CAN"
            total_medals=int(row["total_medals"]),
            avg_snowfall=float(row["avg_snowfall"]),
            log_gdp=float(row["log_gdp"]),
            log_population=float(row["log_population"]),
            won_any=int(row["won_any"]),  # 1 = won at least one medal
            name=str(row["team_name"]),
        )

    # ── Build feature matrix ───────────────────────────────────────────────
    # Keep the list of node IDs aligned with the rows of X_std so that indices[i, j] correctly maps back to a country code.
    countries = list(summary["node_id"])
    feature_cols = ["avg_snowfall", "log_gdp", "log_population"]

    feat_df = summary.set_index("node_id")[feature_cols].loc[countries]
    X_raw = feat_df.values  # shape: (n_countries, 3)

    # Standardise: subtract column mean, divide by column std.
    # After this step each feature has mean=0 and std=1, so Euclidean distance treats all three dimensions equally.
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_raw)

    # ── k-NN edge construction ─────────────────────────────────────────────
    # NearestNeighbors always includes the query point itself as the closest neighbour (distance=0), so we request K+1 neighbours and skip index 0.
    knn = NearestNeighbors(n_neighbors=K_NEIGHBOURS + 1, metric="euclidean")
    knn.fit(X_std)
    distances, indices = knn.kneighbors(X_std)
    # distances[i, k] = Euclidean distance from country i to its k-th neighbour
    # indices[i, k]   = row index of that neighbour in the countries list

    for i, country in enumerate(countries):
        for j_rank in range(1, K_NEIGHBOURS + 1):  # start at 1 to skip self
            j = indices[i, j_rank]  # row index of neighbour
            dist = distances[i, j_rank]  # distance in std space
            neighbour = countries[j]
            weight = 1.0 / (1.0 + dist)  # similarity: [0, 1]

            # Because we iterate over all pairs in both directions, some edges
            # would be added twice.  has_edge() prevents duplicates.
            if not G.has_edge(country, neighbour):
                G.add_edge(country, neighbour,
                           weight=weight,  # used by degree / PageRank / clustering
                           distance=dist)  # used by betweenness / closeness / Dijkstra

    # ── Summary stats ──────────────────────────────────────────────────────
    print(f"\n[GRAPH] Nodes  : {G.number_of_nodes()}")
    print(f"[GRAPH] Edges  : {G.number_of_edges()}")
    print(f"[GRAPH] Is directed : {nx.is_directed(G)}")
    print(f"[GRAPH] Density     : {nx.density(G):.4f}")
    print(f"[GRAPH] Edge def    : {K_NEIGHBOURS}-nearest neighbours in "
          f"(snowfall, log_gdp, log_population) space")

    # Diameter and average path length are only defined for connected graphs.
    # With k-NN edges the graph is almost always connected, but we handle the edge case by reporting stats on the largest connected component (LCC).
    if nx.is_connected(G):
        print(f"[GRAPH] Diameter         : {nx.diameter(G)}")
        print(f"[GRAPH] Avg path length  : "
              f"{nx.average_shortest_path_length(G):.2f}")
    else:
        lcc = G.subgraph(max(nx.connected_components(G), key=len))
        print(f"[GRAPH] Graph not fully connected — largest component: "
              f"{lcc.number_of_nodes()} nodes")
        print(f"[GRAPH] LCC diameter        : {nx.diameter(lcc)}")
        print(f"[GRAPH] LCC avg path length : "
              f"{nx.average_shortest_path_length(lcc):.2f}")

    return G

# ---------------------------------------------------------------------------
# 3. Node-level metrics
# ---------------------------------------------------------------------------
def compute_metrics(G: nx.Graph) -> dict:
    """
    Compute five standard node-level centrality metrics.

    Each metric captures a different aspect of a country's position in the climate/economic similarity network:
      Degree       — total similarity-weighted connections.
                     A high-degree country has many climate/economic peers.
                     Because edges use similarity as weight, high degree = strongly embedded in a dense neighbourhood of similar countries.

      Betweenness  — how often a country lies on the shortest path between two other countries.
                     Uses 'distance' as the path cost so paths prefer hops through climatically similar intermediaries.
                      A high-betweenness country acts as a 'bridge' across otherwise distinct climate zones.

      Closeness    — inverse of the average shortest distance to all other nodes.
                     Again uses 'distance' as the cost.
                     A high-closeness country is centrally positioned in the climate/economic space — it is never far from any other.

      PageRank     — recursive importance: a country is important if important (high-weight) neighbours point to it.
                     Uses 'weight' (similarity) so stronger neighbours contribute more.
                     Alpha=0.85 is the standard damping factor.

      Clustering   — fraction of a country's neighbours that are also connected to each other (weighted version).
                     A high clustering coefficient means the country sits inside a tight local clique of mutually similar nations.

    Note on the weight/distance distinction:
      NetworkX shortest-path functions (betweenness, closeness, Dijkstra) treat the named attribute as a COST to minimise, so we pass 'distance'.
      Aggregation functions (degree, PageRank, clustering) treat it as a STRENGTH to accumulate, so we pass 'weight' (similarity).
    """
    print("\n[METRICS] Computing node-level centrality metrics...")

    # Weighted degree: sum of similarity weights on all incident edges.
    # Higher = the country has many strongly similar neighbours.
    degrees = dict(G.degree(weight="weight"))

    # Betweenness: fraction of all-pairs shortest paths that pass through this node.  'distance' ensures paths prefer short (similar) hops.
    betweenness = nx.betweenness_centrality( G, weight="distance", normalized=True)

    # Closeness: (n-1) / sum_of_shortest_distances_to_all_others. 'distance' keyword tells NetworkX to use distance as path cost.
    closeness = nx.closeness_centrality(G, distance="distance")

    # PageRank: iterative random-walk importance. 'weight' = similarity, so a walk is more likely to follow high-weight (i.e. more similar) edges.
    pagerank = nx.pagerank(G, alpha=0.85, weight="weight")

    # Weighted local clustering coefficient per node.
    clustering = nx.clustering(G, weight="weight")

    # Bundle into a dict for easy downstream access.
    metrics = {
        "degree": degrees,
        "betweenness": betweenness,
        "closeness": closeness,
        "pagerank": pagerank,
        "clustering": clustering,
    }

    # ── Print leaderboards ────────────────────────────────────────────────
    print("\nTop 10 countries by weighted degree most strongly connected in climate/economic space):")
    for country, deg in sorted(degrees.items(), key=lambda x: -x[1])[:10]:
        # ASCII bar chart scaled to 40 characters for readability.
        bar = "█" * min(int(deg * 10), 40)
        print(f"  {country}: {bar} ({deg:.3f})")

    print("\nTop 5 by Betweenness Centrality:")
    for c, s in sorted(betweenness.items(), key=lambda x: -x[1])[:5]:
        print(f"  {c}: {s:.4f}")

    print("\nTop 5 by PageRank:")
    for c, s in sorted(pagerank.items(), key=lambda x: -x[1])[:5]:
        print(f"  {c}: {s:.4f}")

    # Graph-level summary stats.
    avg_clust = nx.average_clustering(G)  # mean clustering across all nodes
    density = nx.density(G)  # |edges| / |possible edges|
    print(f"\nAverage clustering coefficient : {avg_clust:.4f}")
    print(f"Graph density                  : {density:.4f}")

    return metrics


# ---------------------------------------------------------------------------
# 4. Community detection — greedy modularity
# ---------------------------------------------------------------------------
def detect_communities(G: nx.Graph) -> dict:
    """
    Partition the graph into communities using NetworkX's greedy modularity algorithm.

    What is modularity?
      Modularity measures whether edges within communities are denser than would be expected by chance in a random graph with the same degree sequence.
      A score near 1 means very clear community structure; near 0 means the graph is essentially random.

    Interpretation for this project:
      Communities group countries that are similar in snowfall, GDP, and
      population — the three features that define the edges.
      Medal outcomes play no role in forming communities.
      If medal-winning nations cluster heavily into one or two communities, that is visual evidence that climate/economic profile is associated with Olympic success, supporting
      Hypothesis 1 and 2 from the research blueprint.
    """
    print("\n[COMMUNITIES] Running greedy modularity community detection...")

    # greedy_modularity_communities returns a list of frozensets.
    # Convert to plain sets for easier manipulation.
    communities = nx.community.greedy_modularity_communities(G, weight="weight")
    communities = [set(c) for c in communities]

    # Build a flat node→community_index mapping for colouring and downstream use.
    community_map = {}
    for idx, comm in enumerate(communities):
        for node in comm:
            community_map[node] = idx

    # Print a profile of each community, sorted largest-first.
    print(f"  Found {len(communities)} communities")
    for i, comm in enumerate(sorted(communities, key=len, reverse=True)):
        medals = sum(G.nodes[n]["total_medals"] for n in comm)
        snowy = np.mean([G.nodes[n]["avg_snowfall"] for n in comm])
        n_won = sum(G.nodes[n]["won_any"] for n in comm)
        pct_won = 100 * n_won / len(comm)
        print(f"  Community {i + 1}: {len(comm):3d} countries | "
              f"total medals: {medals:5d} | avg snowfall: {snowy:.1f} mm | "
              f"medal winners: {n_won} ({pct_won:.0f}%)")

    # Compute and report the modularity score of the final partition.
    modularity = nx.community.modularity(G, communities, weight="weight")
    print(f"\n  Modularity score: {modularity:.4f}  "
          f"(higher = more distinct communities; >0.3 is typically meaningful)")

    return community_map


# ---------------------------------------------------------------------------
# 5. Path analytics — Dijkstra
# ---------------------------------------------------------------------------
def path_analytics(G: nx.Graph) -> None:
    """
    Find shortest paths between selected country pairs using Dijkstra's algorithm.

    What does 'shortest path' mean in this graph?
      Each edge stores 'distance' = Euclidean distance in standardised (snowfall, log_gdp, log_population) space.
      Dijkstra finds the sequence of countries that minimises total distance, i.e. the chain of countries that are most climatically/economically similar to one another between
      the source and target.

    Why these pairs?
      The pairs are chosen to contrast very different country profiles:
        NOR → IND  : Norway (cold, wealthy, small)  vs  India (warm, large)
        CAN → AUS  : Canada (cold, wealthy)  vs  Australia (warm, similar GDP)
        FIN → BRA  : Finland (snowy)  vs  Brazil (tropical)
        NOR → FIN  : Two similar northern nations — expect a very short path
                     as a sanity check that the algorithm is working correctly.

    What to look for in the output:
      • Path length — a longer total distance means the two countries are
        further apart in climate/economic space.
      • Path nodes  — the intermediate countries reveal which climate/economic
        'zones' connect the two endpoints.  A path NOR → SWE → DEU → IND
        suggests Germany sits in between northern Europe and South Asia in
        this feature space (e.g. via GDP and population).
    """
    print("\n[PATHS] Shortest path analysis (Dijkstra)...")

    # 'distance' is already stored on every edge from build_graph — no additional transformation needed here.
    interesting_pairs = [
        ("NOR", "IND"),  # Norway (snowy, wealthy) vs India (warm, large)
        ("CAN", "AUS"),  # Canada vs Australia
        ("FIN", "BRA"),  # Finland (snowy) vs Brazil (tropical)
        ("NOR", "FIN"),  # Sanity check: two similar cold nations → short path
    ]

    for src, tgt in interesting_pairs:
        # Guard against countries that were dropped during data cleaning.
        if src not in G.nodes or tgt not in G.nodes:
            print(f"  {src} or {tgt} not in graph — skipping")
            continue
        try:
            # nx.dijkstra_path returns the list of node IDs on the optimal route.
            path = nx.dijkstra_path(G, src, tgt, weight="distance")
            # nx.dijkstra_path_length returns the total cost (sum of distances).
            length = nx.dijkstra_path_length(G, src, tgt, weight="distance")
            print(f"\n  {src} → {tgt}:")
            print(f"    Path   : {' → '.join(path)}")
            print(f"    Length : {length:.3f}  "
                  f"(total distance in standardised climate/economic space)")
        except nx.NetworkXNoPath:
            # Can only happen if the graph is not connected.
            print(f"  No path between {src} and {tgt}")

# ---------------------------------------------------------------------------
# 7. Plots
# ---------------------------------------------------------------------------
def plot_graph_communities(G: nx.Graph, community_map: dict, pos: dict) -> None:
    """
    Draw the full country similarity graph, coloured by detected community.

    Visual encoding:
      Node colour  — community membership (one colour per community).
      Node size    — proportional to total medals won across all Games.
                     Larger = more successful at the Winter Olympics.
      Node label   — shown only for countries with more than 5 total medals,
                     to avoid label clutter on the smaller nodes.
      Edge opacity — kept low (0.15) so the graph structure is visible
                     without the edges dominating the visualisation.

    What to look for:
      If medal-rich countries (large nodes) cluster within one or two communities, that visually supports the claim that climate/economic profile is associated with Winter Olympic performance.
    """
    n_comm = len(set(community_map.values()))
    # Assign each node a colour based on its community index.
    color_list = [COLORS[community_map.get(n, 0) % len(COLORS)] for n in G.nodes()]

    # Scale node size linearly between 50 (no medals) and 450 (most medals).
    medal_vals = [G.nodes[n]["total_medals"] for n in G.nodes()]
    max_medals = max(medal_vals) if max(medal_vals) > 0 else 1
    node_sizes = [50 + (m / max_medals) * 400 for m in medal_vals]

    fig, ax = plt.subplots(figsize=(14, 9))
    # Draw edges first so they sit behind the nodes.
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.15,
                           edge_color="#7FA8C0", width=0.5)
    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color=color_list,
                           node_size=node_sizes, alpha=0.9)

    # Only label countries with meaningful medal counts to reduce clutter.
    medal_labels = {n: n for n in G.nodes() if G.nodes[n]["total_medals"] > 5}
    nx.draw_networkx_labels(G, pos, labels=medal_labels, ax=ax,
                            font_size=6, font_color="white")

    # Legend: one patch per community.
    handles = [mpatches.Patch(color=COLORS[i % len(COLORS)],
                              label=f"Community {i + 1}")
               for i in range(min(n_comm, len(COLORS)))]
    ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.3)

    ax.set_title(
        "Winter Olympics Country Graph — Communities detected by greedy modularity\n"
        "Node size = total medals  |  "
        "Edges = climate & economic similarity (snowfall, GDP, population)",
        fontsize=11, pad=12,
    )
    ax.axis("off")
    fig.tight_layout()
    path = FIG_DIR / "G1_olympic_graph.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")


def plot_centrality_maps(G: nx.Graph, metrics: dict, pos: dict) -> None:
    """
    Three-panel figure showing betweenness, closeness, and PageRank as
    node colour intensity on the same graph layout.

    Using the same spring-layout position (pos) across all three panels
    makes it easy to compare which countries score highly on each metric.
    The top 5 nodes per metric are labelled.

    What to look for:
      • Do high-betweenness countries sit geographically between distinct
        climate zones (e.g. mid-latitude countries between polar and tropical)?
      • Do high-PageRank countries overlap with high-medal countries?
        If so, network position in the climate space predicts performance.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    titles = ["Betweenness Centrality", "Closeness Centrality", "PageRank"]
    keys = ["betweenness", "closeness", "pagerank"]
    cmaps = ["Purples", "Blues", "Oranges"]  # distinct colourmap per panel

    for ax, title, key, cmap in zip(axes, titles, keys, cmaps):
        # Extract metric value for every node in graph-node order.
        values = [metrics[key][n] for n in G.nodes()]
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.1,
                               edge_color="#7FA8C0", width=0.4)
        # Colour nodes by their metric value using the chosen colourmap.
        nx.draw_networkx_nodes(G, pos, ax=ax,
                               node_color=values, cmap=cmap,
                               node_size=60, alpha=0.9)
        # Label only the top 5 to avoid clutter.
        top5 = sorted(metrics[key], key=metrics[key].get, reverse=True)[:5]
        nx.draw_networkx_labels(G, pos,
                                labels={n: n for n in top5},
                                ax=ax, font_size=7, font_color="white")
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    fig.suptitle(
        "Node Centrality Maps — Climate/Economic Similarity Network\n"
        "(Edges = snowfall + GDP + population proximity)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    path = FIG_DIR / "G2_centrality_maps.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")


def plot_degree_vs_medals(G: nx.Graph, metrics: dict) -> None:
    """
    Scatter plot: weighted degree in the similarity graph vs total medals.

    In this graph, degree reflects how many climatically/economically similar
    countries surround a given country.  A high-degree country is 'typical'
    — it sits in a dense neighbourhood of peers.  A low-degree country is an
    outlier in climate/economic space (e.g. a small tropical island nation
    with an unusual combination of features).

    The trend line (linear regression) shows whether being more typical
    (higher degree) is correlated with winning more medals overall.

    What to look for:
      • If the trend is positive, climatically/economically 'typical'
        countries tend to win more medals — consistent with the idea that
        countries similar to established Winter Olympic nations do well.
      • If the trend is flat or negative, network position adds little
        beyond the raw node attributes.
    """
    degrees = metrics["degree"]
    nodes = list(G.nodes())
    x = [degrees[n] for n in nodes]  # degree values
    y = [G.nodes[n]["total_medals"] for n in nodes]  # medal counts
    # Colour by medal status: orange = won at least one, blue = none.
    colors = ["#F4A261" if G.nodes[n]["won_any"] else "#7FA8C0" for n in nodes]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(x, y, c=colors, s=60, alpha=0.8,
               edgecolors="white", linewidths=0.4)

    # Annotate the top 10 medal-winning countries for easy identification.
    top10 = sorted(nodes, key=lambda n: G.nodes[n]["total_medals"], reverse=True)[:10]
    for n in top10:
        ax.annotate(f"  {n}", (degrees[n], G.nodes[n]["total_medals"]),
                    fontsize=7, color="#FFD166", fontweight="bold")

    # Fit and draw a linear trend line using numpy's least-squares polyfit.
    m, b = np.polyfit(x, y, 1)  # m = slope, b = intercept
    x_line = np.linspace(min(x), max(x), 100)
    ax.plot(x_line, m * x_line + b, color="#E63946",
            linewidth=1.5, linestyle="--", label=f"Trend (slope={m:.2f})")

    ax.set_xlabel("Weighted Degree (climate/economic similarity to neighbours)")
    ax.set_ylabel("Total Winter Olympic Medals (all years)")
    ax.set_title(
        "Degree in Climate/Economic Network vs Total Medals\n"
        "Orange = won at least one medal  |  Blue = no medals"
    )
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
# Main — orchestrates the full pipeline
# ---------------------------------------------------------------------------
def main():
    """
    Run the complete graph analytics pipeline in order:
      load → build graph → metrics → communities → paths → regression → plots

    The spring layout (pos) is computed once from the graph structure and then
    reused in every plot so that node positions are consistent across all figures.
    """
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load and aggregate country-level data ───────────────────────────
    summary = load_country_summary()

    # ── 2. Build the climate/economic similarity graph ─────────────────────
    G = build_graph(summary)

    # Compute a spring layout once — nodes connected by high-weight edges
    # are pulled closer together; k controls the equilibrium edge length.
    pos = nx.spring_layout(G, seed=42, k=0.4)

    # ── 3. Node-level centrality metrics ──────────────────────────────────
    metrics = compute_metrics(G)

    # ── 4. Community detection ────────────────────────────────────────────
    community_map = detect_communities(G)

    # ── 5. Dijkstra shortest paths between contrasting country pairs ───────
    path_analytics(G)

    # ── 7. Visualisations ─────────────────────────────────────────────────
    print("\n[PLOTS]")
    plot_graph_communities(G, community_map, pos)  # G1: communities
    plot_centrality_maps(G, metrics, pos)  # G2: betweenness/closeness/PageRank
    plot_degree_vs_medals(G, metrics)  # G3: degree vs medals scatter

    print(f"\n✓ Graph analytics complete → figures saved to {FIG_DIR}/")
    print("  G1_olympic_graph.png       — community structure")
    print("  G2_centrality_maps.png     — betweenness / closeness / PageRank")
    print("  G3_degree_vs_medals.png    — degree vs medals scatter")


if __name__ == "__main__":
    main()