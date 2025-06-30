import os
import json
import re
import streamlit as st
import pandas as pd

# ========= Utility: Section Key Normalization ===========
def normalize_section_key(section):
    return (section or "").strip().lower()

def get_breadcrumb(s):
    levels = [s.get(k) for k in ["title", "subtitle", "chapter", "subchapter", "part", "section"] if s.get(k)]
    # Also replace em-dash unicode with ASCII dash for readability
    return " > ".join([str(x).replace("\u2014", "-") for x in levels])

# ========= Load bill and summary ===========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
BILL_JSON = os.path.join(DATA_DIR, "aligned_sections.json")
SUMMARY_JSON = os.path.join(DATA_DIR, "summaries.json")
TITLE_SUMMARY = os.path.join(DATA_DIR, "title_summaries.json")

@st.cache_data
def load_bill_sections(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_summaries(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # get the list of section dicts
    sections = data.get("sections", data if isinstance(data, list) else [])
    # build a mapping from normalized section title → summary text
    return {
        normalize_section_key(s["section"]): 
            s.get("section_summary", s.get("summary", "")) 
        for s in sections 
        if "section" in s
    }
bill_sections = load_bill_sections(BILL_JSON)
section_summaries = load_summaries(SUMMARY_JSON)

@st.cache_data
def load_title_summaries(path):
    import os
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

title_summaries = load_title_summaries(TITLE_SUMMARY)

# ========= Load & clean all CSVs  (unchanged) ================
@st.cache_data
def load_and_clean():

    def clean_label(x: str) -> str:
        return re.sub(r"[^\w\s\-:&/]", "", str(x)).strip()

    def _to_long(fn: str, value_name: str) -> pd.DataFrame:
        tmp = pd.read_csv(fn, dtype=str)
        first, years = tmp.columns[0], tmp.columns[1:]
        rename = {first: "Function"}
        for c in years:
            m = re.search(r"(20\d{2})", c)
            rename[c] = m.group(1) if m else c
        tmp = tmp.rename(columns=rename)
        long = tmp.melt(
            id_vars="Function",
            value_vars=list(rename.values())[1:],
            var_name="FY Year",
            value_name=value_name,
        )
        long["FY Year"] = pd.to_numeric(long["FY Year"], errors="coerce").astype("Int64")
        long[value_name] = pd.to_numeric(long[value_name], errors="coerce")
        long["Function"] = long["Function"].apply(clean_label)
        return long

    HIST2_FN   = os.path.join(DATA_DIR, "hist03z2.csv")
    AUTH_SF_FN = os.path.join(DATA_DIR, "hist05z1.csv")
    T11_FN     = os.path.join(DATA_DIR, "table1-1.csv")
    T12_FN     = os.path.join(DATA_DIR, "table1-2.csv")
    AGENCY_FN  = os.path.join(DATA_DIR, "outlays_by_agency.csv")

    hist_sf = pd.read_csv(HIST2_FN, dtype=str)
    hist_sf["FY Year"]      = pd.to_numeric(hist_sf["year"], errors="coerce")
    hist_sf  = hist_sf.dropna(subset=["FY Year"])
    hist_sf["FY Year"]      = hist_sf["FY Year"].astype("Int64")
    hist_sf["Outlays_Hist"] = pd.to_numeric(hist_sf["amount"], errors="coerce")
    hist_sf = hist_sf.rename(columns={"function": "Function"})
    hist_sf["Function"] = hist_sf["Function"].apply(clean_label)

    auth_sf = pd.read_csv(AUTH_SF_FN, dtype=str)
    auth_sf["FY Year"]            = pd.to_numeric(auth_sf["year"], errors="coerce")
    auth_sf  = auth_sf.dropna(subset=["FY Year"])
    auth_sf["FY Year"]            = auth_sf["FY Year"].astype("Int64")
    auth_sf["Authority_Baseline"] = pd.to_numeric(auth_sf["amount"], errors="coerce")
    auth_sf = auth_sf.rename(columns={"function": "Function"})
    auth_sf["Function"] = auth_sf["Function"].apply(clean_label)

    outlays_fn = _to_long(T11_FN, "Baseline_Outlays_fn")
    auth_fn    = _to_long(T12_FN, "Baseline_Authority_fn")

    ag = pd.read_csv(AGENCY_FN, dtype=str)
    ag = ag[ag["sheet"] == "hist04z1"].copy()
    ag["FY Year"]        = pd.to_numeric(ag["year"], errors="coerce")
    ag   = ag.dropna(subset=["FY Year"])
    ag["FY Year"]        = ag["FY Year"].astype("Int64")
    ag["Outlays_Agency"] = pd.to_numeric(ag["amount"], errors="coerce")
    agency_df = (
        ag[["function", "FY Year", "Outlays_Agency"]]
        .rename(columns={"function": "Agency"})
    )

    return hist_sf, auth_sf, outlays_fn, auth_fn, agency_df

hist_sf_df, auth_sf_df, outlays_fn_df, auth_fn_df, agency_df = load_and_clean()
auth_sf_df = auth_sf_df.drop_duplicates(subset=["Function", "FY Year"])
hist_sf_df = hist_sf_df.drop_duplicates(subset=["Function", "FY Year"])
auth_fn_df = auth_fn_df.drop_duplicates(subset=["Function", "FY Year"])
UNWANTED = hist_sf_df["Function"].str.contains(
    r"^\s*(Total|On-budget|Off-budget|\(On|\(Off)", regex=True
)
hist_sf_df = hist_sf_df[~UNWANTED].copy()
hist_sf_df["Outlays_Hist"] = pd.to_numeric(
    hist_sf_df["Outlays_Hist"], errors="coerce"
)
hist_fn_df = (
    hist_sf_df
      .groupby(["Function", "FY Year"], as_index=False)["Outlays_Hist"]
      .sum()
)
hist_fn_df = hist_fn_df.drop_duplicates(subset=["Function", "FY Year"])

def filter_ui(df: pd.DataFrame, *, label_col: str, key_prefix: str):
    years  = sorted(df["FY Year"].unique())
    labels = sorted(df[label_col].unique())

    sel_years = st.multiselect(
        "Fiscal years", years, years[-5:],
        key=f"{key_prefix}_years"
    )
    sel_labels = st.multiselect(
        label_col, labels, [],
        key=f"{key_prefix}_labels"
    )
    out = df[df["FY Year"].isin(sel_years)]
    if sel_labels:
        out = out[out[label_col].isin(sel_labels)]
    return out

# ==================== Streamlit UI =====================
st.set_page_config(layout="wide")
st.title("FY26 Budget Explorer & Analytics")

tabs = st.tabs([
    "Bill Summaries (by Title)",
    "Bill Explorer",
    "Search Bill",
    "By Function/Subfunction (Historical)",
    "Historic vs Authorized Budget",
    "By Agency (Historical)",
])

# ========== Tab 1: Title Summaries ==========
with tabs[0]:  
    st.header("Synthesized Title Summaries")
    if not title_summaries:
        st.info("No title summaries available yet. Run the synthesis script first.")
    else:
        title_names = [t["title"] for t in title_summaries]
        # Let user select a title to view
        selected_title = st.selectbox("Select a Title", title_names)
        selected_summary = next((t for t in title_summaries if t["title"] == selected_title), None)
        if selected_summary:
            st.subheader(selected_summary["title"])
            st.markdown(f"**Number of sections:** {selected_summary.get('section_count', 'N/A')}")
            st.markdown("### Synthesized Summary")
            st.write(selected_summary.get("synthesized_summary", "_No synthesized summary available._"))
            with st.expander("Show all section summaries under this Title"):
                for s in selected_summary.get("section_summaries", []):
                    st.markdown(f"**{s['section']}**")
                    st.write(s['summary'])
                    st.markdown("---")
        else:
            st.warning("Could not find data for the selected title.")

# --- Bill Explorer: Hierarchical/Tree UI ---
with tabs[1]:
    st.header("Bill Explorer")
    st.markdown("Browse and explore the full legislative text and AI summaries. Expand the hierarchy and click on any section to view its contents.")

    # Build a nested hierarchy: Title > Subtitle > Chapter > Subchapter > Part > Section
    from collections import defaultdict

    def make_hierarchy(sections):
        tree = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))))
        for s in sections:
            title      = s.get("title") or "Other"
            subtitle   = s.get("subtitle")
            chapter    = s.get("chapter")
            subchapter = s.get("subchapter")
            part       = s.get("part")
            key = (title, subtitle, chapter, subchapter, part)
            # Each level may be None; group accordingly
            tree[title][subtitle][chapter][subchapter][part].append(s)
        return tree

    hierarchy = make_hierarchy(bill_sections)

    # Session state for selected section
    if "selected_section_id" not in st.session_state:
        st.session_state.selected_section_id = bill_sections[0]["section"]

    # Render the hierarchy recursively
    def render_sections(sections, indent=0):
        for s in sections:
            sec_id = s["section"]
            label = s["section"]
            # Show the section as a button
            if st.button(label, key=f"section-btn-{sec_id}", help=get_breadcrumb(s)):
                st.session_state.selected_section_id = sec_id

    def render_part(part_dict, indent=0):
        for part, sections in part_dict.items():
            if part:
                with st.expander(f"Part: {part}", expanded=False):
                    render_sections(sections, indent+1)
            else:
                render_sections(sections, indent+1)

    def render_subchapter(subch_dict, indent=0):
        for subch, part_dict in subch_dict.items():
            if subch:
                with st.expander(f"Subchapter: {subch}", expanded=False):
                    render_part(part_dict, indent+1)
            else:
                render_part(part_dict, indent+1)

    def render_chapter(chap_dict, indent=0):
        for chap, subch_dict in chap_dict.items():
            if chap:
                with st.expander(f"Chapter: {chap}", expanded=False):
                    render_subchapter(subch_dict, indent+1)
            else:
                render_subchapter(subch_dict, indent+1)

    def render_subtitle(subt_dict, indent=0):
        for subt, chap_dict in subt_dict.items():
            if subt:
                with st.expander(f"Subtitle: {subt}", expanded=False):
                    render_chapter(chap_dict, indent+1)
            else:
                render_chapter(chap_dict, indent+1)

    # UI: Two columns
    left, right = st.columns([2, 3], gap="large")
    with left:
        for title, subt_dict in hierarchy.items():
            with st.expander(f"Title: {title}", expanded=False):
                render_subtitle(subt_dict, indent=0)

    with right:
        # Find currently-selected section object
        selected_section_obj = next((s for s in bill_sections if s["section"] == st.session_state.selected_section_id), bill_sections[0])
        st.subheader(selected_section_obj["section"])
        st.caption(get_breadcrumb(selected_section_obj))
        st.markdown("**Section Text:**")
        st.text(selected_section_obj.get('text') or "_(No text)_")
        st.markdown("**Summary:**")
        summary = section_summaries.get(normalize_section_key(selected_section_obj["section"]))
        if summary:
            st.text(summary)
        else:
            st.info("No summary available for this section.")

# --- Search Bill ---
with tabs[2]:
    st.header("Search Bill Text & Summaries")
    query = st.text_input("Search by keyword (title, section, or text):")
    if query:
        results = []
        for s in bill_sections:
            if (
                query.lower() in s['section'].lower()
                or query.lower() in s.get('text', '').lower()
                or any(query.lower() in (s.get(k, "") or "").lower() for k in ["title", "subtitle", "chapter", "subchapter", "part"])
            ):
                results.append(s)
        st.write(f"Found {len(results)} sections matching '{query}':")
        for s in results:
            st.markdown(f"**{s['section']}**")
            st.caption(get_breadcrumb(s))
            st.text(s['text'][:2000] + ("..." if len(s['text']) > 2000 else ""))
            summ = section_summaries.get(normalize_section_key(s['section']))
            if summ:
                st.markdown("*Summary:*")
                st.text(summ[:2000] + ("..." if len(summ) > 2000 else ""))
            st.markdown("---")
    else:
        st.info("Enter a keyword above to search the bill.")
# ========== Tab 4 : Historic Outlays (Function/Subfunction) ==========
with tabs[3]:
    st.header("Historic Outlays by Line Item by FY (in millions of dollars)")
    UNWANTED = hist_sf_df["Function"].str.contains(
        r"^\s*(Total|On-budget|Off-budget|\(On|\(Off)", regex=True
    )
    hist_sf_clean = hist_sf_df[~UNWANTED].copy()
    hist_sf_clean["Outlays_Hist"] = pd.to_numeric(
        hist_sf_clean["Outlays_Hist"], errors="coerce"
    )
    hagg = (
        hist_sf_clean
        .groupby(["FY Year", "Function"], as_index=False)["Outlays_Hist"]
        .sum()
    )
    value_col = "Outlays_Hist"
    pivot1 = (
        hagg.pivot(index="FY Year",
                   columns="Function",
                   values=value_col)
             .fillna(0)
    )
    st.line_chart(pivot1, use_container_width=True)

# ========== Tab 5 : Historic vs Authorized Budget ==========
with tabs[4]:
    st.header("Historic vs Authorized Budget")
    view = st.radio(
        "Granularity",
        ["Subfunction view", "Function view"],
        horizontal=True
    )

    DESIRED = ["Function/Subfunction", "FY Year",
               "Outlays_Hist", "Authority_Baseline"]

    if view == "Subfunction view":
        UNWANTED = auth_sf_df["Function"].str.contains(
        r"^\s*(Total|On-budget|Off-budget|\(On|\(Off)", regex=True, na=False
    )
        auth_sf_df_ = auth_sf_df[~UNWANTED].copy()
        df_sf_raw = (
            hist_sf_df
            .merge(auth_sf_df_, on=["Function", "FY Year"], how="outer")
            .fillna(0)
            .rename(columns={
                "Function": "Function/Subfunction"
            })
        )
        df_sf_raw = df_sf_raw[DESIRED]
        df_sf_raw = df_sf_raw.query("Outlays_Hist != 0 or Authority_Baseline != 0")
        df_cmp = filter_ui(
            df_sf_raw,
            label_col="Function/Subfunction",
            key_prefix="subf"
        )
        st.subheader("Funds per subfunction (in millions of dollars)")
        st.dataframe(df_cmp, height=300)
        out_pivot  = df_cmp.pivot_table(index="FY Year",
                                        columns="Function/Subfunction",
                                        values="Outlays_Hist",
                                        aggfunc="sum",
                                        fill_value=0)
        auth_pivot = df_cmp.pivot_table(index="FY Year",
                                        columns="Function/Subfunction",
                                        values="Authority_Baseline",
                                        aggfunc="sum",
                                        fill_value=0)
        st.subheader("Outlays by subfunction (in millions of dollars)")
        st.bar_chart(out_pivot,  use_container_width=True)
        st.subheader("Authorized by subfunction (in millions of dollars)")
        st.bar_chart(auth_pivot, use_container_width=True)
    else:
        UNWANTED = auth_fn_df["Function"].str.contains(
        r"^\s*(Total|On-budget|Off-budget|\(On|\(Off)", regex=True, na=False
    )
        auth_fn_df_ = auth_fn_df[~UNWANTED].copy()
        df_fn_raw = (
            hist_fn_df
            .merge(auth_fn_df_, on=["Function", "FY Year"], how="outer")
            .fillna(0)
            .rename(columns={
                "Function":              "Function/Subfunction",
                "Baseline_Authority_fn": "Authority_Baseline"
            })
        )
        df_fn_raw = df_fn_raw[["Function/Subfunction", "FY Year",
                            "Outlays_Hist", "Authority_Baseline"]]
        df_fn_raw = df_fn_raw.query("Outlays_Hist != 0 or Authority_Baseline != 0")
        fn_cmp = filter_ui(
            df_fn_raw,
            label_col="Function/Subfunction",
            key_prefix="func"
        )
        st.subheader("Funds by function (in millions of dollars)")
        st.dataframe(fn_cmp, height=300)
        out_fn = fn_cmp.pivot_table(index="FY Year",
                                    columns="Function/Subfunction",
                                    values="Outlays_Hist",
                                    aggfunc="sum",
                                    fill_value=0)
        auth_fn = fn_cmp.pivot_table(index="FY Year",
                                    columns="Function/Subfunction",
                                    values="Authority_Baseline",
                                    aggfunc="sum",
                                    fill_value=0)
        st.subheader("Outlays by function")
        st.bar_chart(out_fn,  use_container_width=True)
        st.subheader("Authorized by function")
        st.bar_chart(auth_fn, use_container_width=True)

# ========== Tab 6: By Agency (Historic) ==========
with tabs[5]:
    st.header("Historic Outlays by Agency")
    agency_filt = filter_ui(agency_df, label_col="Agency", key_prefix="agency")
    st.subheader("Agency Outlay Table (in millions of dollars)")
    st.dataframe(agency_filt, height=300)
    years = sorted(agency_filt["FY Year"].dropna().unique())
    target_year = 2025 if 2025 in years else years[-1]
    top10 = (
            agency_filt[agency_filt["FY Year"] == target_year]
            .nlargest(10, "Outlays_Agency")
            .set_index("Agency")["Outlays_Agency"]
        )
    st.subheader(f"Top 10 Agencies (FY{target_year}, in millions of dollars)")
    st.bar_chart(top10, use_container_width=True)






