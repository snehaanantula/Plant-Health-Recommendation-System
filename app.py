from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Plant Health Helper",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.image_utils import ImageInputError, decode_image, image_quality_warnings
from src.predictor import predict_image, read_model_metadata
from src.recommendations import get_recommendation

ROOT = Path(__file__).resolve().parent
MULTICROP_MODEL_DIR = ROOT / "models" / "multicrop"
DEFAULT_MODEL_DIR = ROOT / "models"
ACTIVE_MODEL_DIR = (
    MULTICROP_MODEL_DIR
    if (MULTICROP_MODEL_DIR / "plant_model.keras").is_file()
    and (MULTICROP_MODEL_DIR / "class_names.json").is_file()
    else DEFAULT_MODEL_DIR
)
MODEL_PATH = ACTIVE_MODEL_DIR / "plant_model.keras"
METADATA_PATH = ACTIVE_MODEL_DIR / "class_names.json"
REPORTS_DIR = ROOT / "reports" / "evaluation"
if ACTIVE_MODEL_DIR == MULTICROP_MODEL_DIR:
    REPORTS_DIR = REPORTS_DIR / "multicrop"
METRICS_PATH = REPORTS_DIR / "metrics.json"
CONFUSION_MATRIX_PATH = REPORTS_DIR / "confusion_matrix.png"
TRAINING_HISTORY_PATH = ACTIVE_MODEL_DIR / "training_history.json"
MODEL_INPUT_SIZE = (224, 224)
DEFAULT_SUPPORTED_CLASSES = (
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___healthy",
)
SUPPORTED_CLASSES = tuple(
    read_model_metadata(METADATA_PATH)["class_names"]
    if METADATA_PATH.is_file()
    else DEFAULT_SUPPORTED_CLASSES
)
PLANTVILLAGE_URL = "https://github.com/spMohanty/PlantVillage-Dataset"
BACKGROUND_PATH = ROOT / "assets" / "leaf_detail_optimized.webp"
HERO_BACKGROUND_PATH = ROOT / "assets" / "forest_background_optimized.webp"
LEAF_BACKGROUND_PATH = ROOT / "assets" / "crop_hero_optimized.webp"
LEAF_ILLUSTRATIONS = {
    "Tomato___Early_blight": ROOT / "assets" / "tomato_early_blight.svg",
    "Tomato___Late_blight": ROOT / "assets" / "tomato_late_blight.svg",
    "Tomato___healthy": ROOT / "assets" / "tomato_healthy.svg",
}
background_image = base64.b64encode(BACKGROUND_PATH.read_bytes()).decode("ascii")
hero_background_image = base64.b64encode(HERO_BACKGROUND_PATH.read_bytes()).decode("ascii")
leaf_background_image = base64.b64encode(LEAF_BACKGROUND_PATH.read_bytes()).decode("ascii")

st.markdown(
    """
    <style>
    @keyframes section-enter {
        from {opacity: 0; transform: translateY(8px);}
        to {opacity: 1; transform: translateY(0);}
    }
    .block-container {max-width: 1120px; padding-top: 2rem; padding-bottom: 3rem; scroll-behavior: smooth;}
    [data-testid="stAppViewContainer"] {
        background-image:
            linear-gradient(180deg, rgba(8, 27, 14, .56), rgba(7, 23, 12, .72)),
            url("data:image/webp;base64,__BACKGROUND__");
        background-position: center;
        background-size: cover;
        background-attachment: fixed;
    }
    [data-testid="stMain"], [data-testid="stMainBlockContainer"],
    [data-testid="stHeader"] {background: transparent !important;}
    [data-testid="stMainBlockContainer"] {animation: section-enter .42s ease-out both;}
    [data-testid="stMain"] [data-testid="stMarkdownContainer"],
    [data-testid="stMain"] [data-testid="stWidgetLabel"],
    [data-testid="stSidebar"] * {color: #fff;}
    [data-testid="stSidebar"] {background: rgba(15, 42, 24, .62);}
    [data-testid="stAlert"] {background: rgba(14, 39, 22, .8);}
    [data-testid="stAlert"] [data-testid="stMarkdownContainer"] {color: #fff;}
    .hero {
        padding: 2rem 2.2rem; border-radius: 20px;
        background-image:
            linear-gradient(90deg, rgba(9, 32, 17, .78), rgba(9, 32, 17, .35)),
            url("data:image/webp;base64,__HERO_BACKGROUND__");
        background-position: center 54%; background-size: cover;
        border: 1px solid rgba(255, 255, 255, .32);
        box-shadow: 0 14px 36px rgba(0, 0, 0, .2); margin-bottom: 1.2rem;
        transition: box-shadow .25s ease, transform .25s ease;
    }
    .hero:hover {box-shadow: 0 18px 42px rgba(0, 0, 0, .28); transform: translateY(-2px);}
    .hero h1 {color: #fff; text-shadow: 0 1px 4px #17351f; margin: 0 0 .5rem 0; font-size: 2.35rem;}
    .hero p {color: #fff; text-shadow: 0 1px 3px #17351f; font-size: 1.05rem; margin: 0;}
    .small-note {color: #53645a; font-size: .9rem;}
    .st-key-recommendation_panel, .st-key-disease_information_panel {
        padding: 1rem; border: 1px solid rgba(255, 255, 255, .24);
        border-radius: 18px;
        background-image:
            linear-gradient(135deg, rgba(9, 32, 17, .91), rgba(13, 42, 23, .84)),
            url("data:image/webp;base64,__LEAF_BACKGROUND__");
        background-position: center; background-size: cover;
        box-shadow: 0 10px 28px rgba(0, 0, 0, .16);
        transition: box-shadow .25s ease, transform .25s ease;
    }
    .st-key-recommendation_panel:hover, .st-key-disease_information_panel:hover {
        box-shadow: 0 14px 34px rgba(0, 0, 0, .24);
    }
    .st-key-recommendation_panel [data-testid="stHorizontalBlock"] > div {
        padding: .8rem; border: 1px solid rgba(255, 255, 255, .2);
        border-radius: 14px;
        background-image:
            linear-gradient(145deg, rgba(10, 35, 18, .9), rgba(12, 38, 20, .82)),
            url("data:image/webp;base64,__LEAF_BACKGROUND__");
        background-position: center; background-size: cover;
        transition: transform .2s ease, box-shadow .2s ease;
    }
    .st-key-recommendation_panel [data-testid="stHorizontalBlock"] > div:hover {
        transform: translateY(-2px); box-shadow: 0 8px 20px rgba(0, 0, 0, .2);
    }
    [data-testid="stVerticalBlockBorderWrapper"]:has([class*="st-key-leaf_example_"]) {
        overflow: hidden; border-radius: 18px;
        background-image:
            linear-gradient(145deg, rgba(12, 39, 21, .88), rgba(12, 39, 21, .72)),
            url("data:image/webp;base64,__LEAF_BACKGROUND__");
        background-position: center; background-size: cover;
        box-shadow: 0 10px 26px rgba(0, 0, 0, .18);
        transition: transform .25s ease, box-shadow .25s ease;
    }
    [data-testid="stVerticalBlockBorderWrapper"]:has([class*="st-key-leaf_example_"]):hover {
        transform: translateY(-4px);
        box-shadow: 0 16px 34px rgba(0, 0, 0, .3);
    }
    [class*="st-key-leaf_example_"] [data-testid="stImage"] img {
        border-radius: 12px; transition: transform .3s ease;
    }
    [data-testid="stVerticalBlockBorderWrapper"]:has([class*="st-key-leaf_example_"]):hover
    [class*="st-key-leaf_example_"] [data-testid="stImage"] img {transform: scale(1.025);}
    div.stButton > button {
        transition: transform .2s ease, box-shadow .2s ease, background-color .2s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-2px); box-shadow: 0 7px 18px rgba(0, 0, 0, .2);
    }
    div.stButton > button[kind="primary"] {
        background: #287a42; border-color: #287a42; min-height: 2.8rem;
    }
    [data-testid="stExpander"], [data-testid="stFileUploader"],
    [data-testid="stCameraInput"] {transition: border-color .2s ease, box-shadow .2s ease;}
    [data-testid="stExpander"]:hover, [data-testid="stFileUploader"]:hover,
    [data-testid="stCameraInput"]:hover {box-shadow: 0 6px 18px rgba(0, 0, 0, .16);}
    @media (max-width: 640px) {
        .block-container {padding: 1rem 1rem 2rem;}
        [data-testid="stAppViewContainer"] {background-attachment: scroll;}
        .hero {padding: 1.4rem; background-position: 57% center;}
        .hero h1 {font-size: 1.8rem;}
        .st-key-recommendation_panel, .st-key-disease_information_panel {padding: .75rem;}
    }
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: .01ms !important;
            animation-iteration-count: 1 !important;
            scroll-behavior: auto !important;
            transition-duration: .01ms !important;
        }
    }
    </style>
    """
    .replace("__BACKGROUND__", background_image)
    .replace("__HERO_BACKGROUND__", hero_background_image)
    .replace("__LEAF_BACKGROUND__", leaf_background_image),
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the plant model...")
def load_model(model_path: str):
    from src.predictor import load_keras_model

    return load_keras_model(Path(model_path))


def show_recommendation(label: str) -> None:
    try:
        advice = get_recommendation(label)
    except KeyError:
        plant, condition = class_parts(label)
        with st.container(key="recommendation_panel"):
            st.subheader("General plant-care guidance")
            st.write(
                f"The model matched this image to the {condition} class for "
                f"{plant}. This is an image pattern, not a confirmed diagnosis."
            )
            st.markdown("**Suggested next steps**")
            st.markdown(
                "- Inspect several leaves and the whole plant; compare symptoms "
                "over time.\n"
                "- Keep clear photos and note when symptoms appeared.\n"
                "- Ask a local agricultural extension service or qualified "
                "crop adviser to confirm the cause."
            )
            st.markdown("**Prevention and monitoring**")
            st.markdown(
                "- Follow locally recommended watering, spacing, and sanitation "
                "practices for this crop.\n"
                "- Monitor nearby plants and clean tools between plants."
            )
            st.warning(
                "No crop-specific treatment guidance has been reviewed for this "
                "class. Do not choose a pesticide from this prediction; seek "
                "local expert advice and follow registered product labels."
            )
        return

    with st.container(key="recommendation_panel"):
        st.subheader(advice["title"])
        st.write(advice["summary"])
        left, right = st.columns(2)
        with left:
            st.markdown("**Suggested next steps**")
            for step in advice["steps"]:
                st.markdown(f"- {step}")
        with right:
            st.markdown("**Prevention and monitoring**")
            for tip in advice["prevention"]:
                st.markdown(f"- {tip}")
        st.info(advice["seek_help"])
        st.markdown("**Treatment and product safety**")
        st.write(
            "This photo model cannot prescribe a medicine or pesticide. Confirm "
            "the problem with a local agricultural extension service or qualified "
            "crop adviser before choosing a product. If a product is recommended, "
            "use only one registered for your crop and location and follow its "
            "label exactly. Never mix products based on an app result."
        )
        if advice["sources"]:
            st.link_button(
                "Open trusted crop guidance",
                advice["sources"][0]["url"],
                use_container_width=False,
            )
        with st.expander("Trusted references"):
            for source in advice["sources"]:
                st.markdown(f"- [{source['name']}]({source['url']})")


def show_disease_information(label: str) -> None:
    try:
        advice = get_recommendation(label)
    except KeyError:
        plant, condition = class_parts(label)
        with st.container(key="disease_information_panel"):
            st.subheader("Disease information")
            st.markdown(f"**Plant:** {plant}")
            st.markdown(f"**Model class:** {condition}")
            st.write(
                "PlantVillage class labels describe the category in the training "
                "dataset. This prediction cannot confirm the pathogen or cause; "
                "other stresses may look similar."
            )
            st.caption(
                "No separately reviewed symptom description is available for "
                "this class. Ask a local crop adviser to confirm."
            )
        return

    details = advice["disease_info"]
    with st.container(key="disease_information_panel"):
        st.subheader("Disease information")
        left, right = st.columns(2)
        with left:
            st.markdown(f"**Plant:** {details['plant']}")
            st.markdown(f"**Condition:** {details['disease']}")
            st.write(details["description"])
        with right:
            st.markdown("**Common symptoms in this class**")
            for symptom in details["symptoms"]:
                st.markdown(f"- {symptom}")
            st.markdown("**Possible contributing conditions**")
            for cause in details["causes"]:
                st.markdown(f"- {cause}")
        st.caption(
            "These are general educational descriptions of image patterns, not a "
            "confirmed diagnosis of the pathogen or cause."
        )


def display_class_name(class_name: str) -> str:
    plant, condition = class_parts(class_name)
    return f"{plant} — {condition}"


def class_parts(class_name: str) -> tuple[str, str]:
    plant, separator, condition = class_name.partition("___")
    if not separator:
        return "Plant", class_name.replace("_", " ")
    return plant.replace("_", " "), condition.replace("_", " ").strip()


def select_leaf_example(class_name: str) -> None:
    st.session_state["selected_leaf_example"] = class_name


def render_interactive_examples() -> None:
    st.subheader("Explore tomato leaf patterns")
    st.caption(
        "Select an illustration to explore general information. These are "
        "original educational drawings, not real photographs and not model "
        "predictions."
    )
    example_classes = tuple(LEAF_ILLUSTRATIONS)
    columns = st.columns(3)
    for column, class_name in zip(columns, example_classes):
        with column:
            with st.container(
                border=True,
                key=f"leaf_example_{class_name.lower()}",
            ):
                st.image(
                    LEAF_ILLUSTRATIONS[class_name].read_text(encoding="utf-8"),
                    caption=display_class_name(class_name),
                    use_container_width=True,
                )
                st.markdown(f"**{display_class_name(class_name)}**")
                st.button(
                    "Explore this leaf",
                    key=f"explore_{class_name}",
                    use_container_width=True,
                    on_click=select_leaf_example,
                    args=(class_name,),
                )

    selected = st.session_state.get("selected_leaf_example")
    if selected not in LEAF_ILLUSTRATIONS:
        return

    st.divider()
    image_col, info_col = st.columns([1, 1.25], vertical_alignment="center")
    with image_col:
        st.image(
            LEAF_ILLUSTRATIONS[selected].read_text(encoding="utf-8"),
            use_container_width=True,
        )
    with info_col:
        st.markdown(f"### {display_class_name(selected)}")
        st.caption("Illustration only · Not analyzed by the classifier")
        show_recommendation(selected)
        st.button(
            "Close example",
            key="close_leaf_example",
            on_click=lambda: st.session_state.pop("selected_leaf_example", None),
        )


def clear_analysis() -> None:
    st.session_state["photo_widget_version"] = (
        st.session_state.get("photo_widget_version", 0) + 1
    )
    st.session_state.pop("analysis_result", None)
    st.session_state.pop("analysis_error", None)


def render_analysis() -> None:
    st.markdown(
        """
        <section class="hero">
          <h1>🌿 Plant Health Helper</h1>
          <p>Check a supported crop leaf photo and get a cautious, easy-to-understand
          result with practical next steps.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.warning(
        "For learning and demonstration only—not a confirmed diagnosis. This "
        f"Model classes currently cover {len(SUPPORTED_CLASSES)} crop and "
        "condition combinations. Check Supported plants before use."
    )
    st.link_button(
        "Open PlantVillage dataset",
        PLANTVILLAGE_URL,
        help="View the public dataset repository and its terms.",
    )

    with st.expander("Photo tips for a better result", expanded=True):
        st.markdown(
            "- Photograph one leaf in focus with the affected area visible.\n"
            "- Use bright, even light; avoid shadows, glare, and filters.\n"
            "- Include the whole leaf and keep the camera steady.\n"
            "- Only plant and condition classes listed in Supported plants "
            "are included in this model."
        )

    version = st.session_state.get("photo_widget_version", 0)
    source = st.radio(
        "Choose how to add a photo",
        ("Upload photo", "Take photo"),
        horizontal=True,
        key=f"photo_source_{version}",
    )
    if source == "Upload photo":
        photo = st.file_uploader(
            "Choose a JPG, PNG, or WebP photo (maximum 10 MB)",
            type=("jpg", "jpeg", "png", "webp"),
            key=f"photo_upload_{version}",
        )
    else:
        photo = st.camera_input(
            "Take a supported crop-leaf photo",
            key=f"photo_camera_{version}",
        )

    if photo is None:
        st.caption(
            "Your browser asks for camera permission only if you choose "
            "**Take photo**. You can always use photo upload instead."
        )
        st.session_state.pop("analysis_result", None)
        st.session_state.pop("analysis_error", None)
        return

    image_bytes = photo.getvalue()
    photo_id = hashlib.sha256(image_bytes).hexdigest()
    try:
        image = decode_image(image_bytes)
        quality_warnings = image_quality_warnings(image)
    except (ImageInputError, RuntimeError) as exc:
        st.error(str(exc))
        st.session_state.pop("analysis_result", None)
        return

    for warning in quality_warnings:
        st.warning(warning)

    prior = st.session_state.get("analysis_result")
    if prior and prior.get("photo_id") != photo_id:
        st.session_state.pop("analysis_result", None)
        st.session_state.pop("analysis_error", None)

    st.image(image, caption="Photo preview", use_container_width=True)
    if st.button("Analyze leaf photo", type="primary", use_container_width=True):
        st.session_state.pop("analysis_result", None)
        st.session_state.pop("analysis_error", None)
        if not MODEL_PATH.is_file() or not METADATA_PATH.is_file():
            st.session_state["analysis_error"] = (
                "The trained model is not available yet. Follow the README to "
                "prepare the dataset and train the model."
            )
        else:
            try:
                with st.spinner("Checking the image..."):
                    metadata = read_model_metadata(METADATA_PATH)
                    model = load_model(str(MODEL_PATH))
                    result = predict_image(
                        image=image,
                        model=model,
                        class_names=metadata["class_names"],
                        confidence_threshold=metadata["confidence_threshold"],
                    )
                st.session_state["analysis_result"] = {
                    "photo_id": photo_id,
                    "label": result.label,
                    "confidence": result.confidence,
                    "alternatives": result.alternatives,
                    "display_label": result.display_label,
                }
                history_label = (
                    result.label
                    if result.label is not None
                    else (
                        result.alternatives[0][0]
                        if result.alternatives
                        else "Uncertain"
                    )
                )
                history = st.session_state.setdefault("prediction_history", [])
                history.insert(
                    0,
                    {
                        "time": datetime.now().astimezone().strftime(
                            "%Y-%m-%d %H:%M"
                        ),
                        "plant": class_parts(history_label)[0]
                        if history_label != "Uncertain"
                        else "Unknown",
                        "prediction": display_class_name(history_label)
                        if history_label != "Uncertain"
                        else "Uncertain",
                        "status": (
                            "Uncertain"
                            if result.label is None
                            else (
                                "Healthy"
                                if result.label.endswith("___healthy")
                                else "Diseased"
                            )
                        ),
                        "confidence": result.confidence,
                    },
                )
                del history[50:]
            except (OSError, ValueError, RuntimeError, KeyError) as exc:
                st.session_state["analysis_error"] = (
                    f"Could not analyze this photo: {exc}"
                )

    error = st.session_state.get("analysis_error")
    if error:
        st.error(error)

    result = st.session_state.get("analysis_result")
    if not result or result.get("photo_id") != photo_id:
        return

    st.divider()
    if result["label"] is None:
        st.warning(
            f"The model is not confident enough to suggest a likely condition "
            f"(highest score: {result['confidence']:.0%}). Try a clearer photo "
            "or ask a local plant-health expert. This is not proof the plant "
            "is healthy."
        )
    else:
        is_healthy = result["label"].endswith("___healthy")
        status_text = "HEALTHY-CLASS RESULT" if is_healthy else "DISEASE-CLASS RESULT"
        st.success(
            f"{'🟢' if is_healthy else '🔴'} {status_text} — "
            f"**{result['display_label']}**"
        )
        st.metric(
            "Model score (not diagnostic certainty)",
            f"{result['confidence']:.1%}",
        )
        st.progress(
            min(max(float(result["confidence"]), 0.0), 1.0),
            text="Prediction confidence score",
        )
        st.caption(
            "This score is not a probability that the result is correct. "
            "Image classifiers can be confidently wrong."
        )
        show_disease_information(result["label"])
        st.divider()
        show_recommendation(result["label"])
        if result["alternatives"]:
            with st.expander("Other model scores"):
                for label, score in result["alternatives"]:
                    st.write(f"{label.replace('_', ' ')}: {score:.1%}")

    st.button(
        "Analyze another photo",
        on_click=clear_analysis,
        help="Clear this photo and start a new analysis.",
    )
    st.caption(
        "Photos are processed locally by this app and are not saved by it. "
        "Seek local expert advice for important or spreading problems. This "
        "app does not prescribe pesticides."
    )


def render_demo() -> None:
    st.title("Example result")
    st.info(
        "This is an illustrative UI preview, not a prediction. It does not "
        "analyze a photo or demonstrate model accuracy."
    )
    render_interactive_examples()


def render_how_it_works() -> None:
    st.title("How it works")
    st.write(
        "The app resizes your photo and sends it to a locally saved "
        "MobileNetV2 image classifier. The model compares the image with the "
        "classes it learned during training, then the app looks up general "
        "guidance for the predicted class."
    )
    st.markdown(
        """
        1. Add a clear supported-plant leaf photo by upload or camera.
        2. Preview it and select **Analyze leaf photo**.
        3. The local model scores its trained crop and condition classes.
        4. Low-confidence results are reported as uncertain.
        5. For an accepted class, the app displays general care guidance and
           reference links.
        """
    )
    st.write("**Supported classes**")
    st.markdown("\n".join(f"- {display_class_name(name)}" for name in SUPPORTED_CLASSES))
    st.caption(
        "The app needs a trained model in the models folder. See README.md for "
        "dataset preparation and training instructions."
    )
    st.link_button("View PlantVillage dataset on GitHub", PLANTVILLAGE_URL)
    render_interactive_examples()


def render_history() -> None:
    st.title("Prediction history")
    history = st.session_state.get("prediction_history", [])
    if not history:
        st.info(
            "No image predictions in this browser session yet. Analyze a photo "
            "to add a result here."
        )
        return

    table = pd.DataFrame(history)
    table["confidence"] = table["confidence"].map(lambda value: f"{value:.1%}")
    st.dataframe(
        table,
        hide_index=True,
        use_container_width=True,
        column_config={
            "time": "Time",
            "plant": "Plant",
            "prediction": "Prediction",
            "status": "Status",
            "confidence": "Model score",
        },
    )
    st.caption(
        "History is kept only in Streamlit session state. It is cleared when "
        "the session ends and does not contain uploaded photos."
    )
    st.button(
        "Clear prediction history",
        on_click=lambda: st.session_state.pop("prediction_history", None),
    )


def render_disease_library() -> None:
    st.title("Disease library")
    st.write(
        "Browse the classes included in the trained model. Crop-specific "
        "reviewed guidance is available for selected tomato classes."
    )
    selected = st.selectbox(
        "Choose a plant class",
        SUPPORTED_CLASSES,
        format_func=display_class_name,
    )
    column, image_column = st.columns([1.2, 1])
    with column:
        show_disease_information(selected)
        show_recommendation(selected)
    with image_column:
        if selected in LEAF_ILLUSTRATIONS:
            st.image(
                LEAF_ILLUSTRATIONS[selected].read_text(encoding="utf-8"),
                caption="Educational illustration—not a diagnostic reference photo",
                use_container_width=True,
            )
        else:
            st.image(
                str(LEAF_BACKGROUND_PATH),
                caption="Plant imagery—not a diagnostic reference photo",
                use_container_width=True,
            )


def render_supported_plants() -> None:
    st.title("Supported plants")
    st.write(
        "Only plants and conditions included in the currently trained model "
        "are supported. Predictions are image-class matches, not confirmed "
        "diagnoses."
    )
    classes_by_plant: dict[str, list[str]] = {}
    for class_name in SUPPORTED_CLASSES:
        plant, _ = class_parts(class_name)
        classes_by_plant.setdefault(plant, []).append(class_name)
    st.caption(
        f"{len(classes_by_plant)} crop groups · {len(SUPPORTED_CLASSES)} "
        "trained classes"
    )
    for plant, class_names in classes_by_plant.items():
        st.markdown(f"### {plant}")
        for class_name in class_names:
            st.markdown(f"- {display_class_name(class_name)}")


def render_model_performance() -> None:
    st.title("About the model")
    st.write(
        "The classifier uses MobileNetV2 transfer learning and is trained "
        "offline on the labeled PlantVillage crop classes."
    )
    info_left, info_right = st.columns(2)
    with info_left:
        st.markdown("**Architecture:** MobileNetV2")
        st.markdown("**Technique:** Transfer learning")
        st.markdown("**Dataset source:** PlantVillage")
    with info_right:
        st.markdown(f"**Number of supported classes:** {len(SUPPORTED_CLASSES)}")
        st.markdown(
            f"**Model input size:** {MODEL_INPUT_SIZE[0]} × {MODEL_INPUT_SIZE[1]}"
        )
        st.markdown(
            f"**Crop groups:** {len({class_parts(name)[0] for name in SUPPORTED_CLASSES})}"
        )
    st.link_button("View PlantVillage dataset source", PLANTVILLAGE_URL)

    if not METRICS_PATH.is_file():
        st.info(
            "No measured test metrics are available yet. Train the model and "
            "run `python -m training.evaluate_model` to generate them. This "
            "page does not display invented accuracy values."
        )
    else:
        try:
            metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
            test_accuracy = float(metrics["test_accuracy"])
            macro_f1 = float(metrics["test_macro_f1"])
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            st.error(f"Could not read measured model metrics: {exc}")
        else:
            accuracy_col, f1_col = st.columns(2)
            accuracy_col.metric("Held-out test accuracy", f"{test_accuracy:.1%}")
            f1_col.metric("Held-out macro F1", f"{macro_f1:.1%}")
            st.caption(metrics.get("note", "Metrics are from the held-out dataset test split."))

    if TRAINING_HISTORY_PATH.is_file():
        try:
            training_history = json.loads(
                TRAINING_HISTORY_PATH.read_text(encoding="utf-8")
            )
            accuracy_data = {
                key: values
                for key, values in training_history.items()
                if key in ("accuracy", "val_accuracy")
            }
            loss_data = {
                key: values
                for key, values in training_history.items()
                if key in ("loss", "val_loss")
            }
        except (OSError, json.JSONDecodeError) as exc:
            st.error(f"Could not read training history: {exc}")
        else:
            if accuracy_data and loss_data:
                st.subheader("Training curves")
                accuracy_column, loss_column = st.columns(2)
                with accuracy_column:
                    st.caption("Accuracy")
                    st.line_chart(pd.DataFrame(accuracy_data))
                with loss_column:
                    st.caption("Loss")
                    st.line_chart(pd.DataFrame(loss_data))
    else:
        st.info(
            "Training curves are not available yet. Run the training command "
            "to save them."
        )

    if CONFUSION_MATRIX_PATH.is_file():
        st.subheader("Held-out test confusion matrix")
        st.image(
            str(CONFUSION_MATRIX_PATH),
            caption="Generated by the evaluation script from held-out test images.",
            use_container_width=True,
        )
    else:
        st.info(
            "The confusion matrix is not available yet. Run the evaluation "
            "command after training."
        )


def render_about() -> None:
    st.title("About and limitations")
    st.write(
        "This student project demonstrates computer-vision classification "
        "across the classes listed in Supported plants and a simple "
        "recommendation interface. It is not a diagnostic service."
    )
    st.markdown(
        """
        - It recognizes only classes included in its training data and may
          score unsupported images as one of them.
        - Public training images may differ from real garden or farm photos.
        - Similar symptoms can have different causes; a high score can still
          be wrong.
        - Recommendations are general. Local practices and regulations differ.
        - Do not use this result alone to make consequential crop or treatment
          decisions. Ask a local extension service or qualified crop adviser.
        - Photos are processed by the local app and are not saved by it.
        """
    )
    st.error(
        "No pesticide dosage or product recommendation is provided. Follow "
        "local expert advice and product labels."
    )
    st.link_button("View PlantVillage dataset source", PLANTVILLAGE_URL)


def main() -> None:
    with st.sidebar:
        st.markdown("## 🌿 Plant Health")
        st.caption("Multi-crop leaf photo helper")
        page = st.radio(
            "Navigation",
            (
                "Analyze a leaf",
                "Prediction history",
                "Disease library",
                "Supported plants",
                "Model performance",
                "Example result",
                "How it works",
                "About & limitations",
            ),
            label_visibility="collapsed",
        )
        st.divider()
        st.caption("Educational use only · Local model · No photo storage")

    if page == "Analyze a leaf":
        render_analysis()
    elif page == "Prediction history":
        render_history()
    elif page == "Disease library":
        render_disease_library()
    elif page == "Supported plants":
        render_supported_plants()
    elif page == "Model performance":
        render_model_performance()
    elif page == "Example result":
        render_demo()
    elif page == "How it works":
        render_how_it_works()
    else:
        render_about()


if __name__ == "__main__":
    main()
