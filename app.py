"""
Building Energy Efficiency Predictor
COM763 Advanced Machine Learning

Loads two trained Gradient Boosting pipelines and predicts heating and cooling
loads from valid building configurations represented in the training dataset.
"""

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Building Energy Efficiency Predictor",
    page_icon="🏢",
    layout="centered",
)


@st.cache_resource
def load_models():
    return joblib.load("model.pkl")


try:
    models = load_models()
except FileNotFoundError:
    st.error(
        "The trained model file, model.pkl, could not be found. "
        "Confirm that it is present in the same repository as app.py."
    )
    st.stop()
except Exception as exc:
    st.error(f"The trained models could not be loaded: {exc}")
    st.stop()


BUILDING_SHAPES = [
    {"X1": 0.98, "X2": 514.5, "X3": 294.0, "X4": 110.25, "X5": 7.0},
    {"X1": 0.90, "X2": 563.5, "X3": 318.5, "X4": 122.5, "X5": 7.0},
    {"X1": 0.86, "X2": 588.0, "X3": 294.0, "X4": 147.0, "X5": 7.0},
    {"X1": 0.82, "X2": 612.5, "X3": 318.5, "X4": 147.0, "X5": 7.0},
    {"X1": 0.79, "X2": 637.0, "X3": 343.0, "X4": 147.0, "X5": 7.0},
    {"X1": 0.76, "X2": 661.5, "X3": 416.5, "X4": 122.5, "X5": 7.0},
    {"X1": 0.74, "X2": 686.0, "X3": 245.0, "X4": 220.5, "X5": 3.5},
    {"X1": 0.71, "X2": 710.5, "X3": 269.5, "X4": 220.5, "X5": 3.5},
    {"X1": 0.69, "X2": 735.0, "X3": 294.0, "X4": 220.5, "X5": 3.5},
    {"X1": 0.66, "X2": 759.5, "X3": 318.5, "X4": 220.5, "X5": 3.5},
    {"X1": 0.64, "X2": 784.0, "X3": 343.0, "X4": 220.5, "X5": 3.5},
    {"X1": 0.62, "X2": 808.5, "X3": 367.5, "X4": 220.5, "X5": 3.5},
]

ORIENTATIONS = [2, 3, 4, 5]
GLAZING_AREAS = [0.0, 0.1, 0.25, 0.4]

FEATURES = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]


def format_shape(shape_index: int) -> str:
    shape = BUILDING_SHAPES[shape_index]
    return (
        f"Shape {shape_index + 1}: "
        f"compactness {shape['X1']}, "
        f"surface area {shape['X2']}, "
        f"height {shape['X5']}"
    )


st.title("Building Energy Efficiency Predictor")

st.caption(
    "Estimates the heating and cooling load of a residential building from "
    "design configurations represented in the training dataset. This is a "
    "decision-support prototype, not a substitute for professional "
    "engineering assessment."
)

st.subheader("Building design")

shape_index = st.selectbox(
    "Building shape",
    options=range(len(BUILDING_SHAPES)),
    format_func=format_shape,
    index=3,
    help=(
        "The dataset contains twelve specific building shapes. Relative "
        "compactness, surface area, wall area, roof area and overall height "
        "are therefore selected together rather than independently."
    ),
)

selected_shape = BUILDING_SHAPES[shape_index]

st.caption("Geometry values for the selected building shape")

geometry_summary = pd.DataFrame(
    [
        {
            "Relative compactness": selected_shape["X1"],
            "Surface area": selected_shape["X2"],
            "Wall area": selected_shape["X3"],
            "Roof area": selected_shape["X4"],
            "Overall height": selected_shape["X5"],
        }
    ]
)

st.dataframe(
    geometry_summary,
    use_container_width=True,
    hide_index=True,
)

col1, col2 = st.columns(2)

with col1:
    orientation = st.selectbox(
        "Orientation (dataset code)",
        options=ORIENTATIONS,
        index=2,
        help="Orientation is represented by dataset codes 2 to 5.",
    )

with col2:
    glazing_area = st.selectbox(
        "Glazing area",
        options=GLAZING_AREAS,
        index=2,
    )


if glazing_area == 0.0:
    distribution_options = [0]
    default_distribution_code = 0
    distribution_help = (
        "Glazing distribution is fixed to 0 because the selected glazing "
        "area is 0."
    )
else:
    distribution_options = [1, 2, 3, 4, 5]
    default_distribution_code = 3
    distribution_help = "Non-zero glazing areas use distribution codes 1 to 5."

distribution_index = distribution_options.index(default_distribution_code)

glazing_distribution = st.selectbox(
    "Glazing area distribution (dataset code)",
    options=distribution_options,
    index=distribution_index,
    key=f"glazing_distribution_{glazing_area}",
    help=distribution_help,
)


if st.button("Predict energy loads", type="primary"):
    input_values = {
        **selected_shape,
        "X6": orientation,
        "X7": glazing_area,
        "X8": glazing_distribution,
    }

    input_row = pd.DataFrame([input_values], columns=FEATURES)

    try:
        heating = models["heating"].predict(input_row)[0]
        cooling = models["cooling"].predict(input_row)[0]
    except KeyError:
        st.error(
            "The saved model file does not contain both the heating and "
            "cooling prediction pipelines."
        )
        st.stop()
    except Exception as exc:
        st.error(f"The prediction could not be completed: {exc}")
        st.stop()

    st.subheader("Predicted loads")

    metric1, metric2 = st.columns(2)
    metric1.metric("Heating load", f"{heating:.2f}")
    metric2.metric("Cooling load", f"{cooling:.2f}")

    st.caption(
        "Values are in the same units as the dataset targets. Confirm the "
        "exact unit against the dataset documentation before quoting it."
    )

    st.info(
        "The application restricts geometry and glazing inputs to valid "
        "configurations represented in the training dataset. It is intended "
        "for comparisons within this known simulation grid and should not "
        "be used to assess completely new building forms."
    )
