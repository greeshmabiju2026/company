from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
DATA_FILE = APP_DIR / "G3.csv"
MODEL_FILE = APP_DIR / "company_model.pkl"
DISPLAY_COLUMNS = [
    "display_name",
    "company_type",
    "enterprise_group_jurisdiction",
    "employees",
    "revenue",
    "revenue_currency",
    "market_cap",
    "market_cap_currency",
    "legal_units_count",
    "is_standalone_entity",
]


@st.cache_resource
def load_assets():
    missing_files = [path.name for path in (DATA_FILE, MODEL_FILE) if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(f"Missing required file(s): {', '.join(missing_files)}")

    companies = pd.read_csv(DATA_FILE)
    model = joblib.load(MODEL_FILE)
    features = list(getattr(model, "feature_names_in_", []))
    if not features or len(features) != getattr(model, "n_features_in_", None):
        raise ValueError("The company model does not expose a valid feature schema.")
    missing_columns = set(features) - set(companies.columns)
    if missing_columns:
        raise ValueError(
            "The company CSV is missing model columns: "
            + ", ".join(sorted(missing_columns))
        )
    if "company_type" not in companies:
        raise ValueError("The company CSV is missing the company_type target column.")
    if not hasattr(model, "predict_proba"):
        raise ValueError("The saved company model does not support probability scores.")

    # The original category encoders were not supplied. Alphabetical codes are
    # an explicit fallback so the saved numeric-only model can be explored.
    category_maps = {}
    for column in features:
        if companies[column].dtype == "object":
            categories = sorted(companies[column].dropna().astype(str).unique())
            category_maps[column] = {
                category: code for code, category in enumerate(categories)
            }

    target_labels = sorted(companies["company_type"].dropna().astype(str).unique())
    model_classes = [int(value) for value in model.classes_]
    if model_classes != list(range(len(target_labels))):
        raise ValueError(
            "The model class IDs do not match the sorted company_type labels "
            "in the CSV; the missing target encoder prevents reliable label mapping."
        )

    return companies, model, features, category_maps, target_labels


def prepare_model_input(
    record: pd.Series,
    features: list[str],
    category_maps: dict[str, dict[str, int]],
    companies: pd.DataFrame,
) -> pd.DataFrame:
    values = {}
    for column in features:
        value = record[column]
        if column in category_maps:
            if pd.isna(value):
                values[column] = -1
            else:
                values[column] = category_maps[column].get(str(value), -1)
        elif pd.isna(value):
            median = companies[column].median()
            if pd.isna(median):
                raise ValueError(f"No usable values were found for {column}.")
            values[column] = median
        else:
            values[column] = value
    return pd.DataFrame([values], columns=features)


def format_money(value: float, currency: str | None) -> str:
    if pd.isna(value):
        return "Not reported"
    amount = float(value)
    suffix = f" {currency}" if isinstance(currency, str) and currency else ""
    if abs(amount) >= 1_000_000_000_000:
        return f"{amount / 1_000_000_000_000:,.2f}T{suffix}"
    if abs(amount) >= 1_000_000_000:
        return f"{amount / 1_000_000_000:,.2f}B{suffix}"
    if abs(amount) >= 1_000_000:
        return f"{amount / 1_000_000:,.2f}M{suffix}"
    return f"{amount:,.0f}{suffix}"


st.set_page_config(
    page_title="Enterprise Atlas",
    page_icon="🏢",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(ellipse at 8% 0%, rgba(23, 167, 130, .16), transparent 34%),
            #0d1418;
    }
    .block-container {
        max-width: 1240px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(25, 38, 42, .72);
        border-color: rgba(159, 190, 177, .16);
        border-radius: 18px;
    }
    h1, h2, h3 { letter-spacing: -.025em; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🏢 Enterprise Atlas")
st.markdown(
    "Explore enterprise groups, compare reported company metrics, and try the "
    "saved model’s company-type prediction."
)

try:
    companies, model, features, category_maps, target_labels = load_assets()
except (FileNotFoundError, ImportError, OSError, ValueError) as error:
    st.error(f"Could not load the company data or model: {error}")
    st.stop()

companies = companies.copy()
companies["company_type_display"] = (
    companies["company_type"].fillna("Not reported").astype(str).str.strip()
)
companies["company_type_normalized"] = (
    companies["company_type_display"].str.upper().replace({"": "NOT REPORTED"})
)

with st.sidebar:
    st.header("Filter companies")
    search = st.text_input("Search name or group ID")
    jurisdictions = sorted(
        companies["enterprise_group_jurisdiction"].dropna().astype(str).unique()
    )
    selected_jurisdictions = st.multiselect("Jurisdiction", jurisdictions)
    normalized_types = sorted(companies["company_type_normalized"].unique())
    selected_types = st.multiselect("Company type", normalized_types)

filtered = companies.copy()
if search.strip():
    query = search.strip()
    mask = (
        filtered["display_name"].astype(str).str.contains(query, case=False, regex=False)
        | filtered["enterprise_group_id"]
        .astype(str)
        .str.contains(query, case=False, regex=False)
    )
    filtered = filtered.loc[mask]
if selected_jurisdictions:
    filtered = filtered.loc[
        filtered["enterprise_group_jurisdiction"].isin(selected_jurisdictions)
    ]
if selected_types:
    filtered = filtered.loc[
        filtered["company_type_normalized"].isin(selected_types)
    ]

st.caption(
    f"Showing {len(filtered):,} of {len(companies):,} enterprise groups · "
    "financial amounts retain their reported currencies."
)

metric_columns = st.columns(4)
metric_columns[0].metric("Enterprise groups", f"{len(filtered):,}")
metric_columns[1].metric(
    "Public companies",
    f"{int((filtered['company_type_normalized'] == 'PUBLIC').sum()):,}",
)
median_employees = filtered["employees"].median()
metric_columns[2].metric(
    "Median employees",
    "Not reported" if pd.isna(median_employees) else f"{median_employees:,.0f}",
)
metric_columns[3].metric(
    "Jurisdictions",
    f"{filtered['enterprise_group_jurisdiction'].nunique():,}",
)

overview_tab, explorer_tab, predictor_tab = st.tabs(
    ["📊 Overview", "🔎 Company explorer", "🧠 Model prediction"]
)

with overview_tab:
    chart_columns = st.columns(2)
    with chart_columns[0]:
        type_counts = (
            filtered["company_type_normalized"]
            .value_counts()
            .rename_axis("Company type")
            .reset_index(name="Companies")
        )
        if not type_counts.empty:
            type_chart = px.bar(
                type_counts,
                x="Company type",
                y="Companies",
                color="Company type",
                title="Company type breakdown",
                color_discrete_sequence=px.colors.qualitative.Safe,
            )
            type_chart.update_layout(
                showlegend=False,
                margin=dict(l=12, r=12, t=54, b=12),
            )
            st.plotly_chart(type_chart, use_container_width=True)
        else:
            st.info("No companies match the current filters.")

    with chart_columns[1]:
        currency_options = sorted(
            filtered["revenue_currency"].dropna().astype(str).unique()
        )
        if currency_options:
            selected_currency = st.selectbox("Revenue currency", currency_options)
            currency_companies = filtered.loc[
                (filtered["revenue_currency"] == selected_currency)
                & filtered["revenue"].notna()
            ].nlargest(12, "revenue")
            if not currency_companies.empty:
                revenue_chart = px.bar(
                    currency_companies.sort_values("revenue"),
                    x="revenue",
                    y="display_name",
                    orientation="h",
                    title=f"Top reported revenue · {selected_currency}",
                    labels={"revenue": f"Revenue ({selected_currency})", "display_name": ""},
                    color_discrete_sequence=["#21b891"],
                )
                revenue_chart.update_layout(margin=dict(l=12, r=12, t=54, b=12))
                st.plotly_chart(revenue_chart, use_container_width=True)
            else:
                st.info(f"No revenue figures are available in {selected_currency}.")
        else:
            st.info("No revenue figures are available for the current filters.")

    st.subheader("Reported company metrics")
    metric_data = filtered[["employees", "revenue", "net_income", "market_cap"]].rename(
        columns={
            "employees": "Employees",
            "revenue": "Revenue",
            "net_income": "Net income",
            "market_cap": "Market cap",
        }
    )
    metric_data = metric_data.agg(["count", "median", "mean"]).T
    metric_data["count"] = metric_data["count"].map(lambda value: f"{value:,.0f}")
    for statistic in ["median", "mean"]:
        metric_data[statistic] = metric_data[statistic].map(
            lambda value: f"{value:,.0f}" if pd.notna(value) else "—"
        )
    st.dataframe(metric_data, use_container_width=True)
    st.caption(
        "Financial metrics can use different currencies and reporting dates. "
        "The table shows raw reported figures; it does not aggregate currencies."
    )

with explorer_tab:
    st.subheader("Enterprise group directory")
    if filtered.empty:
        st.info("No companies match the current filters.")
    else:
        display = filtered[DISPLAY_COLUMNS].rename(
            columns={
                "display_name": "Company",
                "company_type": "Company type",
                "enterprise_group_jurisdiction": "Jurisdiction",
                "employees": "Employees",
                "revenue": "Revenue",
                "revenue_currency": "Currency",
                "market_cap": "Market cap",
                "market_cap_currency": "Market cap currency",
                "legal_units_count": "Legal units",
                "is_standalone_entity": "Standalone",
            }
        )
        st.dataframe(
            display,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Employees": st.column_config.NumberColumn(format="%.0f"),
                "Revenue": st.column_config.NumberColumn(format="%.2e"),
                "Market cap": st.column_config.NumberColumn(format="%.2e"),
            },
        )

        chosen_company = st.selectbox(
            "View a company profile",
            filtered["display_name"].tolist(),
            key="profile_company",
        )
        profile = filtered.loc[filtered["display_name"] == chosen_company].iloc[0]
        st.markdown(f"### {profile['display_name']}")
        profile_columns = st.columns(4)
        profile_columns[0].metric(
            "Company type",
            profile["company_type_display"],
        )
        profile_columns[1].metric(
            "Employees",
            "Not reported"
            if pd.isna(profile["employees"])
            else f"{profile['employees']:,.0f}",
        )
        profile_columns[2].metric(
            "Revenue",
            format_money(profile["revenue"], profile["revenue_currency"]),
        )
        profile_columns[3].metric(
            "Market cap",
            format_money(profile["market_cap"], profile["market_cap_currency"]),
        )
        st.write(
            f"**Jurisdiction:** {profile['enterprise_group_jurisdiction']} · "
            f"**Enterprise group ID:** `{profile['enterprise_group_id']}` · "
            f"**Legal units:** {profile['legal_units_count']:,}"
        )
        if isinstance(profile.get("website"), str) and profile["website"].strip():
            website = profile["website"].strip()
            website_url = website if website.startswith(("https://", "http://")) else f"https://{website}"
            st.markdown(f"[Company website]({website_url})")

with predictor_tab:
    st.subheader("Explore a company-type prediction")
    st.warning(
        "The supplied model expects 26 numeric inputs, but its original "
        "text-category encoders and target-label encoder were not included. "
        "This demo uses alphabetical category codes from G3.csv and sorted "
        "company-type labels. Treat predictions as exploratory, not as verified "
        "model outputs."
    )
    st.caption(
        "Pick a company record to see how the saved model scores its type. "
        "Missing numeric fields are filled with the dataset median; missing "
        "text categories use -1."
    )

    prediction_names = companies["display_name"].sort_values().tolist()
    selected_name = st.selectbox("Company to score", prediction_names)
    selected_record = companies.loc[companies["display_name"] == selected_name].iloc[0]
    st.write(
        f"Recorded type: **{selected_record['company_type_display']}** · "
        f"Jurisdiction: **{selected_record['enterprise_group_jurisdiction']}**"
    )

    if st.button("Score company type", type="primary"):
        try:
            model_input = prepare_model_input(
                selected_record,
                features,
                category_maps,
                companies,
            )
            predicted_class = int(model.predict(model_input)[0])
            if not 0 <= predicted_class < len(target_labels):
                raise ValueError("The model returned an unknown company-type class.")
            probabilities = model.predict_proba(model_input)[0]
            ranked = sorted(
                zip(target_labels, probabilities),
                key=lambda entry: entry[1],
                reverse=True,
            )
        except (IndexError, TypeError, ValueError) as error:
            st.error(f"Company scoring failed: {error}")
        else:
            predicted_label = target_labels[predicted_class]
            result_columns = st.columns([1, 2])
            with result_columns[0]:
                st.success(f"Model class: **{predicted_label}**")
                st.metric("Top model score", f"{max(probabilities):.1%}")
                if predicted_label.casefold() == str(
                    selected_record["company_type"]
                ).casefold():
                    st.info("This matches the type recorded in the CSV.")
                else:
                    st.caption(
                        f"CSV label: {selected_record['company_type_display']}"
                    )
            with result_columns[1]:
                scores = pd.DataFrame(
                    {
                        "Model score (%)": [
                            float(probability) * 100
                            for _, probability in ranked
                        ]
                    },
                    index=[label for label, _ in ranked],
                )
                st.bar_chart(scores)
            st.caption(
                "Scores are model probabilities, not necessarily calibrated "
                "confidence estimates."
            )

with st.expander("About the supplied model"):
    st.write(
        f"The saved classifier is a Random Forest with {len(features)} input "
        f"features and {len(target_labels)} output classes. Because the "
        "original encoding files were not provided, categorical preprocessing "
        "and the class-to-label mapping cannot be verified."
    )
    st.write("Model input columns:", ", ".join(features))
    st.write(
        "Inferred output labels:",
        ", ".join(f"{index}: {label}" for index, label in enumerate(target_labels)),
    )
