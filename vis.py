import os
import glob
import json
import re
from collections import defaultdict, Counter
import pandas as pd
import squarify
import numpy as np
import circlify
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import plotly.graph_objects as go
from matplotlib.backends.backend_pdf import PdfPages
# import plotly.io as pio
import matplotlib.colors as mcolors
def abbreviate_label(label: str) -> str:
    # use first letters of words including - , e.g AGENTS ACTING ON THE RENIN-ANGIOTENSIN SYSTEM-> AATARS
    if " " not in label and "-" not in label and "," not in label:
        return label[:4].upper()
    words = re.split(r"[\s\-,]+", label) # including - and , as separators
    print(words)
    # use whole word if its one word
    if len(words) <= 1:
        return words[0][:4].upper()
    abbrev = ""
    for index, word in enumerate(words[:3]):
        if len(word) <= 3 and word.isdigit() == False:
            abbrev +=  word
            abbrev += "_"
            if index == 0:
                abbrev = word
                break
        elif word:
            abbrev +=  word[0]
            abbrev += "_"
    # remove brackets
    abbrev = re.sub(r"[\(\)\[\]\{\}{_}]", "", abbrev)
    return abbrev.upper()
def _norm_text(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    s = re.sub(r"\s+", " ", s)
    # remove surrounding quotes that appear in your output
    s = s.strip('"').strip("'")
    return s

STATUS_ABBREV = {
    "identical match": "IM",
    "compatible match": "CM",
    "partial match (proximate)": "PMP",
    "partial match (tentative)": "PMT",
    "not applicable": "NA",
}
ORDERED_STATUSES = [
    "IM","CM","PMP","PMT","NA"
]

def lighten_color(color, amount=0.85):
    """
    Lighten the given color by mixing it with white.
    amount=1.0 -> white
    amount=0.0 -> original color
    """
    try:
        c = mcolors.to_rgb(color)
    except ValueError:
        c = (0.8, 0.8, 0.8)
    return tuple((1 - amount) * v + amount for v in c)

# CATEGORY_COLORS = {
#     "drug_exposure": "#7E6CB2",
#     "condition_occurrence": "#E15759",
#     "measurement": "#59A14F",
#     "procedure_occurrence": "#EDC948",
#     "observation": "#B07AA1",
#     "specimen": "#EDB6BB",  
#     "device_exposure": "#76B7B2",
#     "visit_occurrence": "#F28E2B",
#     "death": "#9C755F",
#     "person": "#F0E1DA",
# }   

CATEGORY_COLORS = {
    "drug_exposure": "#7E6CB2",
    "condition_occurrence": "#E15759",
    "measurement": "#59A14F",
    "procedure_occurrence": "#EDC948",
    "observation": "#B07AA1",
    "specimen": "#EDB6BB",  
    "device_exposure": "#76B7B2",
    "visit_occurrence": "#F28E2B",
    "death": "#9C755F",
    "person": "#F0E1DA",
}   
def status_abbrev(status: str) -> str:
    return STATUS_ABBREV.get(status.lower(), status[:2].upper())


def build_common_pairs_tree_counts(
    data_dir: str,
    files_pattern: str = "*.csv",
    category_col: str = "category",
    status_col: str = "harmonization_status",
    slabel_col: str = "slabel",
    tlabel_col: str = "tlabel",
) -> dict:
    """
    Returns:
      dict[category][status][pair_label] = count_across_files
    """
    csv_files = sorted(glob.glob(os.path.join(data_dir, files_pattern)))
    print(f"Found {len(csv_files)} CSV files in {data_dir}")

    tree = defaultdict(lambda: defaultdict(Counter))

    for fp in csv_files:
        df = pd.read_csv(fp)

        # guard: skip files missing required columns
        missing = {category_col, status_col, slabel_col, tlabel_col} - set(df.columns)
        if missing:
            print(f"Skipping {os.path.basename(fp)} (missing columns: {sorted(missing)})")
            continue
        # timepoint_based_pairs =  # if while row
        for _, row in df.iterrows():
            category = _norm_text(row.get(category_col, "")) or "unknown_category"
            status = _norm_text(row.get(status_col, "")) or "unknown_status"
            slabel = _norm_text(row.get(slabel_col, ""))
            tlabel = _norm_text(row.get(tlabel_col, ""))

            if not slabel and not tlabel:
                continue
            if category == "condition_era":
                category = "condition_occurrence"
            pair = slabel if (slabel and slabel == tlabel) else f"{slabel} ⟷ {tlabel}".strip()
            if tree[category][status][pair] == 0:
                tree[category][status][pair] += 1

    # convert to normal dicts for JSON friendliness
    return {
        cat: {stat: dict(counter) for stat, counter in stats.items()}
        for cat, stats in tree.items()
    }
    
def abbrev_two_letters(label: str) -> str:
    """
    Example:
      'beta adrenergic receptor antagonist' -> 'BE AD RE AN'
    """
    words = re.split(r"[\s\-_/]+", label.lower())
    return "_".join(w[:2].upper() for w in words if len(w) >= 2)

def build_category_status_concept_hierarchy(tree_counts):
    """
    Returns:
      dict[category][status][concept] = count
    """
    hierarchy = defaultdict(lambda: defaultdict(Counter))

    for category, status_dict in tree_counts.items():
        if category == "condition_era":
            category = "condition_occurrence"
        for status, pairs in status_dict.items():
            for pair, count in pairs.items():
                concepts = re.split(r"\s*⟷\s*", pair)
                for c in concepts:
                    c = c.strip()
                    if c:
                        hierarchy[category][status][c] += count
    print(f"Built hierarchy for {hierarchy} categories")
    return hierarchy


def _adaptive_text_color(rgba) -> str:
    r, g, b, _ = rgba
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if luminance < 0.5 else "black"

def safe_fontsize(circle_radius, n_chars, min_fs=6, max_fs=14):
    """
    Estimate a font size that fits inside a circle.
    """
    # diameter in data coords → heuristic scale
    base = circle_radius * 40

    # penalize long text
    size = base / max(1.0, n_chars / 10)

    return int(np.clip(size, min_fs, max_fs))

def plot_category_packed_circles(
    category: str,
    status_concepts: dict,
    out_dir: str,
):
    """
    One circle per harmonization status.
    Circle size = number of unique concepts in that status.
    Abbreviation inside circle, full label in legend.
    """

    # ---- build data for circlify ----
    data = []
    for status, concepts in status_concepts.items():
        if not concepts:
            continue
        data.append({
            "id": status,
            "datum": len(concepts),  # unique concepts
        })

    if not data:
        return

    # ---- deterministic status order & colors ----
    statuses = sorted(d["id"] for d in data)
    colors = plt.cm.Set2.colors

    status_color_map = {
        status: colors[i % len(colors)]
        for i, status in enumerate(statuses)
    }

    # ---- circle packing ----
    circles = circlify.circlify(
        data,
        show_enclosure=False,
        target_enclosure=circlify.Circle(0, 0, 1),
    )

    # ---- plotting ----
    fig, ax = plt.subplots(figsize=(7, 7), dpi=220)
    ax.set_title(f"{category.upper()}: Harmonization Status Distribution", fontsize=12)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)

    for c in circles:
        if c.ex is None:
            continue

        status = c.ex["id"]
        print(f"Plotting status '{status}' with {len(status_concepts.get(status, {}))} concepts")
        x, y, r = c.x, c.y, c.r
        color = status_color_map[status]

        # draw circle
        ax.add_patch(
            plt.Circle(
                (x, y),
                r,
                facecolor=color,
                edgecolor="black",
                alpha=0.45,
                linewidth=1.2,
            )
        )

        # abbreviated label inside
        label = str(len(status_concepts.get(status, {}))) +'\n'+ status_abbrev(status)
        fontsize = safe_fontsize(r, len(label))

        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            weight="bold",
            clip_on=True,
        )

    # ---- legend (full labels) ----
    legend_handles = [
        Patch(
            facecolor=color,
            edgecolor="black",
            label=f"{status_abbrev(status)} = {status}"
        )
        for status, color in status_color_map.items()
    ]

    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=2,
        frameon=False,
        fontsize=9,
    )

    # ---- save ----
    # out_png = os.path.join(
    #     out_dir,
    #     f"packed_status_{category.replace(' ', '_')}.png"
    # )

    # plt.savefig(out_png, bbox_inches="tight")
    # plt.close(fig)

    # print(f"Saved: {out_png}")
    return fig


def plot_category_status_heatmap(
    tree_counts: dict,
    out_png: str,
    metric: str = "unique_pairs",  # or "total_occurrences"
    figsize=(10, 6),
    dpi: int = 220,
):
    categories = sorted(tree_counts.keys())
    statuses = sorted({s for cat in tree_counts for s in tree_counts[cat].keys()})

    M = np.zeros((len(categories), len(statuses)), dtype=int)

    for i, cat in enumerate(categories):
        for j, stat in enumerate(statuses):
            pairs = tree_counts.get(cat, {}).get(stat, {})
            if metric == "total_occurrences":
                M[i, j] = int(sum(pairs.values()))
            else:
                M[i, j] = int(len(pairs))

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    cmap = plt.get_cmap("YlGnBu")  # light-friendly palette
    im = ax.imshow(M, aspect="auto", cmap=cmap)  # readable + colorblind-friendly

    ax.set_xticks(range(len(statuses)))
    ax.set_xticklabels(statuses, rotation=35, ha="right")
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories)

    # annotate counts
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, str(M[i, j]), ha="center", va="center", fontsize=6)

    ax.set_title(f"Concepts Pair by Category × Harmonization Status)")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)

    plt.tight_layout()
    # fig.savefig(out_png, bbox_inches="tight")
    # plt.close(fig)
    return fig
    print(f"Saved heatmap: {out_png}")


def visualize_common_codes_per_category_per_status(data_dir: str):
    tree_counts = build_common_pairs_tree_counts(data_dir)

   # fetch unique concepts for each category-status
    category_status_concepts = defaultdict(lambda: defaultdict(set))
    for category, status_dict in tree_counts.items():
        for status, pairs_dict in status_dict.items():
            for pair_label in pairs_dict.keys():
                concepts = re.split(r"\s*⟷\s*", pair_label)
                for concept in concepts:
                    category_status_concepts[category][status].add(concept)
    # we need to visualize for one category unique concepts per status using treemap
    for category, status_dict in category_status_concepts.items():
        # prepare data for treemap
        treemap_data = []
        for status, concepts in status_dict.items():
            treemap_data.append({
                "status": status,
                "unique_concepts_count": len(concepts)
            })
        df_treemap = pd.DataFrame(treemap_data)

        # plot treemap
        fig, ax = plt.subplots(figsize=(8, 6), dpi=220)
        squarify.plot(
            sizes=df_treemap["unique_concepts_count"],
            label=[f"{row['status']}\n{row['unique_concepts_count']}" for _, row in df_treemap.iterrows()],
            alpha=0.7,
            color=plt.cm.Paired.colors[:len(df_treemap)],
            ax=ax
        )
        ax.set_title(f"Unique Concepts by Status for Category: {category}")
        plt.axis('off')
        out_png = os.path.join(data_dir, f"treemap_unique_concepts_{category.replace(' ', '_')}.png")
        # plt.savefig(out_png, bbox_inches="tight")
        # plt.close(fig)
        # print(f"Saved treemap: {out_png}")
        return fig

def dataframe_to_pdf_pages(
    df: pd.DataFrame,
    title: str,
    category_col: str = "category",
    max_rows: int = 25,
):
    """
    Split a DataFrame into multiple table figures.
    Rows are color-coded by category.
    
    Returns:
        List[matplotlib.figure.Figure]
    """
    figures = []
    df.fillna(0, inplace=True)
    if df.empty:
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        ax.set_title(title, fontsize=12)
        figures.append(fig)
        return figures

    n_pages = int(np.ceil(len(df) / max_rows))

    for page_idx in range(n_pages):
        start = page_idx * max_rows
        end = start + max_rows
        df_page = df.iloc[start:end]

        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")

        table = ax.table(
            cellText=df_page.values,
            colLabels=df_page.columns,
            loc="center",
            cellLoc="left",
        )

        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.25)

        # ---- header styling ----
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor("#EAEAEA")
                cell.set_text_props(weight="bold")

        # ---- row coloring by category ----
        if category_col in df_page.columns:
            cat_idx = df_page.columns.get_loc(category_col)

            for i, (_, row_data) in enumerate(df_page.iterrows(), start=1):
                category = row_data[category_col]
                base_color = CATEGORY_COLORS.get(category.lower(), "#DDDDDD")
                bg_color = lighten_color(base_color, amount=0.88)

                for col in range(len(df_page.columns)):
                    table[(i, col)].set_facecolor(bg_color)

        page_title = f"{title} (page {page_idx + 1} / {n_pages})"
        ax.set_title(page_title, fontsize=12, pad=14)

        figures.append(fig)

    return figures



def visualize_unique_concepts_across_categories(
    data_dir: str,
    min_label_count: int = 30,   # show labels only for dominant concepts
):
    """
    Treemap of unique clinical concepts across categories.
    Color = category
    Size = frequency across mappings
    """

    tree_counts = build_common_pairs_tree_counts(data_dir)

    records = []

    for category, status_dict in tree_counts.items():
        if category == "condition_era":
            category = "condition_occurrence"

        concept_counter = Counter()
        # add count per harmonization status as well in records
        harmonization_status_counts = defaultdict(Counter)
        
        print(f"status_dict: {status_dict}")
        for status, pairs in status_dict.items():
            harmonization_count = 0
            for pair, count in pairs.items():
                harmonization_count += count
                for concept in re.split(r"\s*⟷\s*", pair):
                    concept = concept.strip()
                    if concept:
                        concept_counter[concept] += count
                        status_ = status_abbrev(status)
                        harmonization_status_counts[concept][status_] += count
        record_item = {}
        for concept, count in concept_counter.items():
            record_item = {
                "category": category,
                "concept": concept,
                "count": int(count),
            }
            # sort harmonization status by ORDERED_STATUSES
            # harmonization_status_counts = dict(
            #     sorted(
            #         harmonization_status_counts.items(),
            #         key=lambda x: ORDERED_STATUSES.index(x[0]) if x[0] in ORDERED_STATUSES else len(ORDERED_STATUSES)
            #     )
            # )
            for status, status_count in harmonization_status_counts[concept].items():
                record_item.update({
                    status:int(status_count)

                })
            records.append(record_item)

    if not records:
        print("No concepts found.")
        return

    df = pd.DataFrame(records)
    
    # # all float to int
    # for col in df.select_dtypes(include=['float']).columns:
    #     df[col] = df[col].astype(int)
        
    df_figs = dataframe_to_pdf_pages(df, title="Unique Clinical Concepts Across Categories", max_rows=25)
    df.to_csv(
        os.path.join(data_dir, "unique_concepts_across_categories.csv"),
        index=False,
    )
    # ---- colors per rectangle ----
    df["color"] = df["category"].map(
        lambda c: CATEGORY_COLORS.get(c, "#DDDDDD")
    )

    # ---- labels: only for large rectangles ----
    def concept_label(row):
        
        return f"{abbreviate_label(row['concept'])}\n({row['count']})"
        # return ""

    labels = df.apply(concept_label, axis=1)
    # labels = labels.tolist()
    # labels = [abbreviate_label(l) if l else "" for l in labels]
    # ---- plot ----
    fig, ax = plt.subplots(figsize=(20, 12), dpi=220)

    squarify.plot(
        sizes=df["count"],
        label=labels,
        color=df["color"],
        alpha=0.85,
        pad=True,
        ax=ax,
    )

    ax.set_title(
        "Unique Clinical Concepts Across Categories",
        fontsize=8,
        pad=10,
    )
    ax.axis("off")

    # ---- category legend ----
    legend_handles = [
        Patch(
            facecolor=color,
            edgecolor="black",
            label=category
        )
        for category, color in CATEGORY_COLORS.items()
        if category in df["category"].unique()
    ]

    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        frameon=False,
        fontsize=9,
        title="Category",
    )

    out_png = os.path.join(
        data_dir,
        "treemap_unique_concepts_by_category.png"
    )
    # plt.savefig(out_png, bbox_inches="tight")
    # plt.close(fig)
    return fig, df_figs
    # print(f"Saved treemap: {out_png}")



def visualize_common_codes_tree(data_dir: str):
    tree_counts = build_common_pairs_tree_counts(data_dir)

    # overview for paper
    plot_category_status_heatmap(
        tree_counts,
        out_png=os.path.join(data_dir, "category_status_heatmap.png"),
        metric="unique_pairs",
    )
    
 
    # optional: also dump the counts tree
    with open(os.path.join(data_dir, "common_pairs_tree_counts.json"), "w", encoding="utf-8") as f:
        json.dump(tree_counts, f, indent=2, ensure_ascii=False)

    return tree_counts

def visualize_packed_circles_per_category(data_dir: str):
    tree_counts = build_common_pairs_tree_counts(data_dir)
    hierarchy = build_category_status_concept_hierarchy(tree_counts)

    for category, status_concepts in hierarchy.items():
        plot_category_packed_circles(
            category=category,
            status_concepts=status_concepts,
            out_dir=data_dir,
        )


def pdf_page_1_report():
    # explain what is in the report
    # studies includes list of studies
    # harmonization method summary
     # Used data dictionaries from X studies: list of studies
    # harmonization summary includes:
    # 1. heatmap of category vs harmonization status
    # 2. treemap of unique concepts across categories
    # 3. packed circles per category of unique concepts per harmonization status
    text = """
    What We Mean by Harmonization in This Report \n 
    In this report, harmonization refers to the process of systematically aligning heterogeneous clinical data elements across multiple cohort studies so that they can be meaningfully compared, queried, and reused for downstream analysis. Importantly, harmonization here is performed at the metadata level, not by transforming or merging patient-level data. The goal is to establish semantic and contextual compatibility between variables originating from different studies, even when their original representations differ in naming, coding, granularity, or structure.
    Harmonization therefore answers the question:
        To what extent do variables from different studies represent the same or compatible clinical concept, under comparable context and usage?
    1. Scope of Harmonization
    This report uses data dictionaries from multiple clinical studies, each describing study variables with information such as:
      a. Variable label and description
      b. Clinical domain (e.g., condition, medication, measurement, procedure)
      c. Units, categorical values, and visit/time context
      d. Mappings to controlled vocabularies (e.g., SNOMED CT, LOINC, ATC, OMOP)
    No assumptions are made about uniform data collection protocols across studies. Differences in study design, population, visit schedules, and variable definitions are explicitly acknowledged and handled.
    2. How Harmonization Is Achieved
        Harmonization is performed through a multi-step, rule-based and semantic process, combining controlled vocabularies, ontology structure, and contextual constraints.
    2.1 . Conceptual Alignment Using Standard Vocabularies
        Each variable is linked (where possible) to standardized clinical concepts using established terminologies such as:
        SNOMED CT for conditions and clinical findings
        LOINC for measurements and observations
        ATC / RxNorm for medications
        OMOP concept identifiers as a unifying reference layer
    These mappings provide a shared semantic anchor across studies.
    2.2. Context-Aware Matching
            Variables are not compared solely by code or label. Harmonization explicitly considers:
            Clinical domain (e.g., measurement vs procedure vs condition)
            Temporal or visit context (e.g., baseline, follow-up, prior to baseline)
            Units of measurement
            Categorical value semantics
            Additional contextual qualifiers (e.g., method, body position, assessment type)
    This prevents false matches such as aligning a procedure date with a measurement value or conflating clinically distinct concepts with similar labels.
    3. Semantic Proximity and Hierarchical Reasoning
            When exact equivalence is not possible, semantic relationships are explored using:
            Hierarchical relations (e.g., parent–child concepts)
            Textual similarity of labels and descriptions
            Domain-specific compatibility rules: Controlled relaxation of granularity (e.g., drug class vs specific agent)
        This allows identification of compatible or partially compatible variables rather than forcing strict equivalence.
    4. Harmonization Status Assignment
        Each pair of variables is assigned a harmonization status reflecting the quality and reliability of alignment, such as: Exact or identical match, Compatible but not identical, Partial or context-dependent match, Not Applicable
    These statuses are used throughout the report to summarize harmonization outcomes.
    How to Read the Visual Summaries in This Report
    The visualizations on subsequent pages provide a high-level overview of harmonization results:
    Heatmap (Category × Harmonization Status)
    Shows how well different clinical domains harmonize across studies.
    Treemap of Unique Concepts Across Categories
    Highlights the diversity and concentration of standardized concepts available per domain.
    Packed Circles per Category
    Illustrates how unique clinical concepts distribute across harmonization statuses, making gaps and strengths immediately visible.
    Together, these views allow analysts to quickly assess:
    Which domains are well aligned
    Where semantic heterogeneity remains
    Which concepts are broadly reusable across studies
    Key Takeaway
    Harmonization in this report is not a simplistic code-matching exercise. It is a context-aware semantic alignment process designed to respect clinical meaning, study design differences, and real-world data heterogeneity—providing a robust foundation for downstream data integration, federated analysis, and reproducible research.
        
    ."""
    # generate a PDF report using above text as first page
    # with PdfPages(out_pdf) as pdf:
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.text(0.5, 0.5, text, ha="center", va="center", wrap=True, fontsize=10)
    ax.set_title("Harmonization Report Overview", fontsize=16, pad=20)
    # fig.savefig(out_pdf, bbox_inches="tight")
    return fig
    
def visualize_common_codes_tree(data_dir: str):
    tree_counts = build_common_pairs_tree_counts(data_dir)
    visualize_packed_circles_per_category(data_dir)
    # overview for paper
    plot_category_status_heatmap(
        tree_counts,
        out_png=os.path.join(data_dir, "category_status_heatmap.png"),
        metric="unique_pairs",
    )

    # visualize_common_codes_per_category_per_status(data_dir)
    visualize_unique_concepts_across_categories(data_dir)  # ← NEW
    # optional: also dump the counts tree
    with open(os.path.join(data_dir, "common_pairs_tree_counts.json"), "w", encoding="utf-8") as f:
        json.dump(tree_counts, f, indent=2, ensure_ascii=False)

    return tree_counts

def generate_pdf_report(data_dir: str, out_pdf: str):
    tree_counts = build_common_pairs_tree_counts(data_dir)

    with PdfPages(out_pdf) as pdf:

        # intro_fig = pdf_page_1_report()
        # pdf.savefig(intro_fig)
        # 1. Heatmap
        
        fig = plot_category_status_heatmap(
            tree_counts,
            out_png=os.path.join(data_dir, "category_status_heatmap.png"),
        )
        pdf.savefig(fig)
        plt.close(fig)
        
         # 2. Global treemap
        fig, df_figs = visualize_unique_concepts_across_categories(
            data_dir
        )
        pdf.savefig(fig)
        plt.close(fig)
      

        # 3. Packed circles per category
        hierarchy = build_category_status_concept_hierarchy(tree_counts)
        for category, status_concepts in hierarchy.items():
            fig = plot_category_packed_circles(
                category=category,
                status_concepts=status_concepts,
                out_dir=data_dir
            )
            pdf.savefig(fig)
            
        for df_fig in df_figs:
            pdf.savefig(df_fig)
            plt.close(df_fig)
       

       
    print(f"PDF report saved: {out_pdf}")

def visualize_variables_per_study_grouped(data_dir: str):
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    records = []

    for fp in csv_files:
        study_id = os.path.basename(fp).replace(".csv", "")
        df = pd.read_csv(fp)
        df.columns = [c.lower() for c in df.columns]

        if "domain" not in df.columns:
            continue
        # lower case domain values
        df["domain"] = df["domain"].str.lower()
        # replace condition era with condition occurrence
        df["domain"] = df["domain"].replace("condition_era", "condition_occurrence")
        # replace observation_period with observation
        df["domain"] = df["domain"].replace("observation_period", "observation")
         # replace device_exposure with device
        counts = df["domain"].value_counts()
        for domain, cnt in counts.items():
            records.append({
                "study": study_id,
                "domain": domain,
                "count": int(cnt),
            })

    if not records:
        print("No valid data found.")
        return

    df = pd.DataFrame(records)

    # ---- ordering (VERY important for readability) ----
    domain_order = (
        df.groupby("domain")["count"]
        .sum()
        .sort_values(ascending=False)
        .index
    )
    study_order = (
        df.groupby("study")["count"]
        .sum()
        .sort_values(ascending=False)
        .index
    )

    df["domain"] = pd.Categorical(df["domain"], categories=domain_order, ordered=True)
    df["study"] = pd.Categorical(df["study"], categories=study_order, ordered=True)

    # ---- plotting ----
    domains = domain_order
    studies = study_order
    x = np.arange(len(domains))
    bar_width = 0.8 / len(studies)

    fig, ax = plt.subplots(figsize=(14, 6), dpi=300)

    for i, study in enumerate(studies):
        subset = (
            df[df["study"] == study]
            .set_index("domain")
            .reindex(domains)
        )

        subset["count"] = subset["count"].fillna(0)
        ax.bar(
            x + i * bar_width,
            subset["count"],
            width=bar_width,
            label=study,
        )

    ax.set_xticks(x + bar_width * (len(studies) - 1) / 2)
    ax.set_xticklabels(domains, rotation=30, ha="right")
    ax.set_ylabel("Number of variables")
    ax.set_xlabel("Clinical domains (OMOP)")
    ax.set_title("Distribution of variables across domains and studies")

    legend = ax.legend(
        title=f"Study\n",
        frameon=False,
        ncol=1,  
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )
    legend.get_title().set_ha("left")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    out_png = os.path.join(data_dir, "variables_per_domain_grouped.png")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close(fig)

    print(f"Saved: {out_png}")
if __name__ == "__main__":
    data_dir = "/Users/komalgilani/Documents/GitHub/CohortVarLinker/data/output/cross_mapping/sapbert"
    visualize_common_codes_tree(data_dir)
    generate_pdf_report(
        data_dir,
        out_pdf=os.path.join(data_dir, "harmonization_summary.pdf"),
    )
    visualize_variables_per_study_grouped("/Users/komalgilani/Documents/GitHub/CohortVarLinker/data/alignment_article_cohorts")
import os
