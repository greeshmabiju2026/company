from html import escape
from pathlib import Path
from string import Template

import joblib
import pandas as pd
import sklearn
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
DATA_FILE = APP_DIR / "G3.csv"
MODEL_FILE = APP_DIR / "company_model.pkl"
STYLE_FILE = APP_DIR / "static" / "style.css"
PROFILE_TEMPLATE = APP_DIR / "templates" / "company_profile.html"
METRICS_TEMPLATE = APP_DIR / "html" / "company_metrics.html"

FEATURE_COLUMNS = [
    "employees",
    "employees_year",
    "revenue",
    "revenue_year",
    "net_income",
    "net_income_year",
    "market_cap",
    "legal_units_count",
    "direct_subsidiaries_count",
    "max_hierarchy_depth",
]
TABLE_COLUMNS = [
    "display_name",
    "company_type",
    "enterprise_group_jurisdiction",
    "employees",
    "revenue",
    "revenue_currency",
    "market_cap",
    "market_cap_currency",
    "legal_units_count",
]


@st.cache_resource
def load_assets():
    missing_files = [
        path.name
        for path in (
            DATA_FILE,
            MODEL_FILE,
            STYLE_FILE,
            PROFILE_TEMPLATE,
            METRICS_TEMPLATE,
        )
        if not path.is_file()
    ]
    if missing_files:
        raise FileNotFoundError(f"Missing required file(s): {', '.join(missing_files)}")

    companies = pd.read_csv(DATA_FILE)
    model = joblib.load(MODEL_FILE)
    features = list(getattr(model, "feature_names_in_", []))
    if not features or len(features) != getattr(model, "n_features_in_", None):
        raise ValueError("The company model does not expose a valid feature schema.")
    if features != FEATURE_COLUMNS:
        raise ValueError("The company model feature schema is not the expected schema.")
    missing_columns = set(features + ["company_type"]) - set(companies.columns)
    if missing_columns:
        raise ValueError(
            "The company CSV is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )
    non_numeric = [
        column
        for column in features
        if not pd.api.types.is_numeric_dtype(companies[column])
    ]
    if non_numeric:
        raise ValueError(
            "The model requires numeric company fields: " + ", ".join(non_numeric)
        )
    if not hasattr(model, "predict_proba"):
        raise ValueError("The saved company model does not support probability scores.")

    medians = companies[features].median()
    if medians.isna().any():
        unusable = medians[medians.isna()].index.tolist()
        raise ValueError(
            "No usable values were found for model fields: " + ", ".join(unusable)
        )

    data_classes = set(companies["company_type"].dropna().astype(str))
    model_classes = {str(value) for value in model.classes_}
    if model_classes != data_classes:
        raise ValueError("The model classes do not match company_type labels in G3.csv.")

    return (
        companies,
        model,
        features,
        medians,
        STYLE_FILE.read_text(encoding="utf-8"),
        Template(PROFILE_TEMPLATE.read_text(encoding="utf-8")),
        Template(METRICS_TEMPLATE.read_text(encoding="utf-8")),
    )


def display_value(value, digits: int = 0) -> str:
    if pd.isna(value):
        return "Not reported"
    if isinstance(value, (int, float)):
        return f"{value:,.{digits}f}"
    return str(value)


def format_money(value, currency) -> str:
    if pd.isna(value):
        return "Not reported"
    suffix = f" {currency}" if isinstance(currency, str) and currency else ""
    amount = float(value)
    for divisor, unit in (
        (1_000_000_000_000, "T"),
        (1_000_000_000, "B"),
        (1_000_000, "M"),
    ):
        if abs(amount) >= divisor:
            return f"{amount / divisor:,.2f}{unit}{suffix}"
    return f"{amount:,.0f}{suffix}"


def safe_website_url(value) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    website = value.strip()
    if website.startswith(("https://", "http://")):
        return website
    return f"https://{website}"


def prepare_model_input(
    record: pd.Series, features: list[str], medians: pd.Series
) -> pd.DataFrame:
    values = {}
    for column in features:
        value = record[column]
        values[column] = medians[column] if pd.isna(value) else value
    return pd.DataFrame([values], columns=features, dtype=float)


st.set_page_config(page_title="Enterprise Atlas", page_icon="🏢", layout="wide")

try:
    companies, model, features, medians, css, profile_template, metrics_template = load_assets()
except (FileNotFoundError, ImportError, OSError, ValueError) as error:
    st.error(f"Could not load the company data or model: {error}")
    st.stop()

st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
if sklearn.__version__ != "1.6.1":
    st.warning(
        "This model was saved with scikit-learn 1.6.1, but the current runtime "
        f"uses {sklearn.__version__}. Install the pinned version in "
        "`requirements.txt` for model compatibility."
    )
companies = companies.copy()
companies["company_type_display"] = (
    companies["company_type"].fillna("Not reported").astype(str).str.strip()
)
companies["company_type_filter"] = (
    companies["company_type_display"].str.upper().replace({"": "NOT REPORTED"})
)

st.title("Enterprise Atlas")
st.markdown(
    "Explore enterprise groups, compare reported company metrics, and score "
    "company records with the supplied random-forest model."
)

with st.sidebar:
    st.header("Filter companies")
    search = st.text_input("Search company or group ID")
    jurisdictions = sorted(
        companies["enterprise_group_jurisdiction"].dropna().astype(str).unique()
    )
    selected_jurisdictions = st.multiselect("Jurisdiction", jurisdictions)
    company_types = sorted(companies["company_type_filter"].unique())
    selected_types = st.multiselect("Company type", company_types)

filtered = companies
if search.strip():
    query = search.strip()
    matches = (
        filtered["display_name"].astype(str).str.contains(query, case=False, regex=False)
        | filtered["enterprise_group_id"].astype(str).str.contains(
            query, case=False, regex=False
        )
    )
    filtered = filtered.loc[matches]
if selected_jurisdictions:
    filtered = filtered.loc[
        filtered["enterprise_group_jurisdiction"].isin(selected_jurisdictions)
    ]
if selected_types:
    filtered = filtered.loc[filtered["company_type_filter"].isin(selected_types)]

st.caption(
    f"Showing {len(filtered):,} of {len(companies):,} enterprise groups. "
    "Financial values retain their reported currencies."
)
summary = st.columns(4)
summary[0].metric("Enterprise groups", f"{len(filtered):,}")
summary[1].metric(
    "Public companies", f"{int((filtered['company_type_filter'] == 'PUBLIC').sum()):,}"
)
median_employees = filtered["employees"].median()
summary[2].metric(
    "Median employees",
    "Not reported" if pd.isna(median_employees) else f"{median_employees:,.0f}",
)
summary[3].metric(
    "Jurisdictions", f"{filtered['enterprise_group_jurisdiction'].nunique():,}"
)

overview_tab, directory_tab, predictor_tab = st.tabs(
    ["Overview", "Company directory", "Model prediction"]
)

with overview_tab:
    chart_columns = st.columns(2)
    with chart_columns[0]:
        type_counts = (
            filtered["company_type_filter"]
            .value_counts()
            .rename_axis("Company type")
            .rename("Companies")
        )
        if type_counts.empty:
            st.info("No companies match the current filters.")
        else:
            st.subheader("Company types")
            st.bar_chart(type_counts)

    with chart_columns[1]:
        currencies = sorted(
            filtered["revenue_currency"].dropna().astype(str).unique()
        )
        if currencies:
            currency = st.selectbox("Revenue currency", currencies)
            top_revenue = filtered.loc[
                (filtered["revenue_currency"] == currency) & filtered["revenue"].notna()
            ].nlargest(12, "revenue")
            if top_revenue.empty:
                st.info(f"No reported revenue is available in {currency}.")
            else:
                st.subheader(f"Top reported revenue · {currency}")
                st.bar_chart(
                    top_revenue.set_index("display_name")["revenue"].sort_values()
                )
        else:
            st.info("No revenue figures are available for the current filters.")

    st.subheader("Reported company metrics")
    metric_data = filtered[["employees", "revenue", "net_income", "market_cap"]].agg(
        ["count", "median", "mean"]
    ).T
    metric_data.index.name = "Metric"
    metric_data = metric_data.rename(
        columns={"count": "Reported values", "median": "Median", "mean": "Mean"}
    )
    st.dataframe(metric_data, width="stretch")
    st.caption(
        "Amounts may use different currencies and reporting dates. Values are "
        "not converted or aggregated across currencies."
    )

with directory_tab:
    st.subheader("Enterprise group directory")
    if filtered.empty:
        st.info("No companies match the current filters.")
    else:
        table = filtered[TABLE_COLUMNS].rename(
            columns={
                "display_name": "Company",
                "company_type": "Company type",
                "enterprise_group_jurisdiction": "Jurisdiction",
                "employees": "Employees",
                "revenue": "Revenue",
                "revenue_currency": "Revenue currency",
                "market_cap": "Market cap",
                "market_cap_currency": "Market cap currency",
                "legal_units_count": "Legal units",
            }
        )
        st.dataframe(table, hide_index=True, width="stretch")

        selected_id = st.selectbox(
            "Company profile",
            filtered["enterprise_group_id"].tolist(),
            format_func=lambda identifier: (
                f"{filtered.loc[filtered['enterprise_group_id'] == identifier, 'display_name'].iloc[0]}"
                f" · {identifier}"
            ),
            key="directory_company",
        )
        profile = companies.loc[
            companies["enterprise_group_id"] == selected_id
        ].iloc[0]
        website_url = safe_website_url(profile["website"])
        website_link = (
            f'<a href="{escape(website_url, quote=True)}" target="_blank" '
            'rel="noopener noreferrer">Visit company website</a>'
            if website_url
            else ""
        )
        profile_html = profile_template.substitute(
            name=escape(str(profile["display_name"])),
            company_type=escape(str(profile["company_type_display"])),
            jurisdiction=escape(str(profile["enterprise_group_jurisdiction"])),
            group_id=escape(str(profile["enterprise_group_id"])),
            employees=escape(display_value(profile["employees"])),
            revenue=escape(format_money(profile["revenue"], profile["revenue_currency"])),
            market_cap=escape(
                format_money(profile["market_cap"], profile["market_cap_currency"])
            ),
            website_link=website_link,
        )
        st.html(f"<style>{css}</style>{profile_html}")

with predictor_tab:
    st.subheader("Score a company record")
    st.caption(
        "The model uses ten numeric company metrics. Missing values are filled "
        "with the corresponding dataset median."
    )
    selected_name_id = st.selectbox(
        "Company to score",
        companies["enterprise_group_id"].tolist(),
        format_func=lambda identifier: (
            f"{companies.loc[companies['enterprise_group_id'] == identifier, 'display_name'].iloc[0]}"
            f" · {identifier}"
        ),
        key="prediction_company",
    )
    record = companies.loc[
        companies["enterprise_group_id"] == selected_name_id
    ].iloc[0]
    st.write(
        f"Recorded type: **{record['company_type_display']}** · "
        f"Jurisdiction: **{record['enterprise_group_jurisdiction']}**"
    )
    if st.button("Predict company type", type="primary"):
        try:
            model_input = prepare_model_input(record, features, medians)
            probabilities = model.predict_proba(model_input)[0]
            ranked = sorted(
                zip(model.classes_, probabilities),
                key=lambda item: item[1],
                reverse=True,
            )
        except (IndexError, TypeError, ValueError) as error:
            st.error(f"Company scoring failed: {error}")
        else:
            predicted_label, top_probability = ranked[0]
            results = st.columns([1, 2])
            with results[0]:
                st.success(f"Predicted company type: **{predicted_label}**")
                st.metric("Top model probability", f"{top_probability:.1%}")
                if str(predicted_label).casefold() == str(
                    record["company_type"]
                ).casefold():
                    st.info("This matches the type recorded in the CSV.")
                else:
                    st.caption(f"CSV label: {record['company_type_display']}")
            with results[1]:
                metrics_html = metrics_template.substitute(
                    rows="".join(
                        "<tr><td>{}</td><td>{:.1%}</td></tr>".format(
                            escape(str(label)), float(probability)
                        )
                        for label, probability in ranked
                    )
                )
                st.html(f"<style>{css}</style>{metrics_html}")
            st.caption(
                "Probabilities are model scores and may not be calibrated confidence "
                "estimates. The dataset includes a few rare or inconsistent labels."
            )

with st.expander("About the data and model"):
    st.write(
        f"The supplied Random Forest accepts {len(features)} numeric metrics and "
        f"predicts {len(model.classes_)} company-type labels. The model's class labels "
        "were checked against the labels in G3.csv."
    )
    st.write("Model input fields:", ", ".join(features))
    st.write(
        "Data quality note: company type labels include case variants and a small "
        "number of rare values; the model preserves those labels as provided."
    )
