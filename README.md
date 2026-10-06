# Plant Health Recommendation System

A student demonstration that classifies crop-leaf photos into labeled
PlantVillage classes and displays conservative, general next-step guidance.
It is not a disease diagnosis tool and does not recommend pesticides.

## Scope

The included PlantVillage dataset has 38 classes across 14 crop groups:
apple, blueberry, cherry, corn, grape, orange, peach, pepper, potato,
raspberry, soybean, squash, strawberry, and tomato. Training discovers the
class folders in the selected dataset directory. The original three tomato
classes are:

- `Tomato___Early_blight`
- `Tomato___Late_blight`
- `Tomato___healthy`

The model can identify only the classes included in its trained metadata. It
cannot identify other crops, conditions, pests, nutrient problems, or
unrelated images reliably. A softmax score is not diagnostic certainty.

The app includes upload/camera input, image quality warnings, healthy/disease
status, a confidence bar with low-confidence abstention, condition information,
general management and prevention guidance, a session-only prediction history,
a disease library, supported-plants page, and model performance page. History
does not retain the uploaded images. The app does not generate pesticide
prescriptions or use Gemini.

## 1. Check your Python environment

Use Python 3.11 in one environment for dependency installation, training, tests,
and running the application. The project environment checked during scaffolding
reported Python 3.11.8 and did not have Streamlit installed, even though a
different installation may exist elsewhere on the computer.

In PowerShell, from this project directory:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python --version
```

If `.venv` already exists and is the environment you intend to use, activate
that environment rather than creating it again. Check that `python --version`
shows the version you intend and that `python -m pip show streamlit tensorflow`
reports installed packages. For Python 3.11.9 specifically, install/select that
runtime first; Python 3.11.8 is not 3.11.9.

TensorFlow's package is large. On Windows, this setup provides CPU-based
training/inference; GPU acceleration is not required for the demo.

## 2. Obtain public data

The full labeled PlantVillage dataset is already extracted under
`data/raw/plantvillage/`. Its source is the public
[PlantVillage-Dataset GitHub repository](https://github.com/spMohanty/PlantVillage-Dataset).
Check upstream licensing and attribution requirements before redistribution;
details are in [data/raw/README.md](data/raw/README.md).

Arrange the selected images like this:

```text
data/
└── raw/
    └── plantvillage/
        ├── Apple___Apple_scab/
        │   ├── image-001.jpg
        │   └── ...
        ├── Potato___Early_blight/
        │   └── ...
        └── Tomato___healthy/
            └── ...
```

Folder spelling and capitalization must match the PlantVillage class names.
Training skips folders with fewer than seven images. Do not commit downloaded images or
generated models to Git. Dataset images from the same plant or near-duplicates
should not be split across train and test sets; the starter script uses a
random stratified split and does not automatically detect duplicates or plant
identity, so inspect and document this limitation.

## 3. Train and evaluate

From the project root:

```powershell
python -m training.train_model --epochs 3 --batch-size 16 --max-images-per-class 200 --models-dir models/multicrop
python -m training.evaluate_model --models-dir models/multicrop --reports-dir reports/evaluation/multicrop
```

The per-class cap gives a quicker, balanced baseline; omit it to use every
available labeled image. Training uses MobileNetV2's pretrained ImageNet
weights and may take a while on a CPU. Lower the batch size (for example,
`--batch-size 8`) if memory is limited. The training command creates a
multi-crop model and metadata under `models/multicrop/`, which the app selects
automatically the next time it starts after training completes:

- `models/multicrop/plant_model.keras` — trained classifier.
- `models/multicrop/class_names.json` — output class order and initial confidence
  threshold.
- `models/multicrop/split_manifest.csv` — reproducible file/split list for evaluation.

Evaluation writes `reports/evaluation/multicrop/metrics.json` and
`reports/evaluation/multicrop/confusion_matrix.png`, then updates the app's
confidence threshold using validation predictions. Review validation coverage and held-out
test metrics before presenting the model. If the validation set cannot reach the
target precision, the script uses a conservative threshold of 0.95; that does
not guarantee the resulting predictions are reliable.

Training also saves `models/multicrop/training_history.json` for the training/validation
accuracy and loss charts. Existing models trained before this feature need to be
trained again to generate those curves. The app shows measured test metrics
only when the evaluation output exists; otherwise it explains that metrics are
not available instead of showing example values.

## 4. Run the app

After training and evaluation:

```powershell
streamlit run app.py
```

The browser offers upload and user-initiated camera capture. Camera permission
is controlled by the browser; image upload remains available when there is no
camera or permission is denied. Hosted camera access normally requires HTTPS.
The app processes the selected image in the local Python process and does not
save it to disk.

## 5. Test

```powershell
python -m unittest discover -s tests -v
```

The tests exercise prediction thresholding, metadata validation, and the
recommendation catalogue without needing a trained model. End-to-end image
inference and model quality still require installing dependencies, downloading
the dataset, training, and trying representative leaf and non-leaf images.

## Project layout

```text
app.py                       Streamlit interface
assets/                      Original interactive tomato-leaf SVG illustrations
src/image_utils.py           Image validation and RGB conversion
src/predictor.py             Model loading, metadata, prediction/abstention
src/recommendations.py       Reviewed guidance lookup
data/recommendations.json    Guidance and reference links by model class
training/train_model.py      Dataset split and MobileNetV2 fine-tuning
training/evaluate_model.py   Held-out metrics, plot, threshold selection
tests/                       Focused standard-library unit tests
models/                      Generated model/metadata/split manifest (ignored)
reports/evaluation/          Generated metrics and plot (ignored)
```

The **Example result** and **How it works** pages include selectable
healthy/early-blight-like/late-blight-like illustrations. These are original
educational drawings, not dataset photographs or predictions, and are not fed
to the classifier.

## Safety, privacy, and known limitations

- This is a classroom prototype, not agricultural advice or a confirmed
  diagnosis. Consult a local extension service or qualified crop adviser for
  important or spreading problems.
- A classifier always scores its known classes. A low-confidence threshold can
  abstain, but cannot reliably detect every unsupported plant or unrelated
  image. The app says so rather than claiming robust unknown-image detection.
- PlantVillage images may have controlled backgrounds and lighting. Performance
  on field photographs can be substantially worse.
- The threshold is based on a small validation split and should be treated as a
  demonstration choice, not a safety guarantee. The displayed score is not
  calibrated diagnostic probability.
- The starter split is stratified and seeded, but does not group by plant or
  automatically eliminate duplicate images.
- Recommendations are general and conservative; crop practices and product
  rules vary by location. The app does not prescribe pesticides.
- The app does not prescribe medicines or pesticides. Confirm the problem
  locally before choosing a product; use only products registered for the crop
  and location and follow the product label exactly.
- The application does not persist uploaded/captured photos. The dataset and
  trained model are local files and are excluded from Git.

Recommendation references are provided in `data/recommendations.json`; review
them and the dataset terms before a public release.
