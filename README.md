# VisionCare AI 👁️
## Deep Learning-Based Diabetic Retinopathy Severity Classification Using Retinal Fundus Images

> **Academic Focus:** Deep Learning for Computer Vision | Convolutional Neural Networks, Transfer Learning, Fine-Tuning, Medical Image Preprocessing, Class Imbalance Handling, Explainable AI (Grad-CAM), and Full-Stack Clinical Deployment.

---

## 🎯 1. Project Overview & Clinical Objective

**VisionCare AI** is an end-to-end deep learning system designed for the automated detection and severity grading of **Diabetic Retinopathy (DR)** from digital ocular fundus photographs. 

Diabetic retinopathy is a leading cause of preventable blindness among working-age adults worldwide. Early microvascular lesions—such as microaneurysms, blot hemorrhages, and intraretinal microvascular abnormalities (IRMA)—often remain asymptomatic until irreversible visual loss occurs. This project applies **EfficientNetB3 transfer learning** and **fine-tuning** to automate clinical grading in accordance with the International Clinical Diabetic Retinopathy Disease Severity Scale, providing visual explainability through **Grad-CAM (Gradient-weighted Class Activation Mapping)** and generating downloadable patient screening reports.

---

## 🔬 2. Dataset Specification: APTOS 2019 Blindness Detection

The model is trained and evaluated on the benchmark **APTOS 2019 Blindness Detection** dataset ([Kaggle Competition Source](https://www.kaggle.com/competitions/aptos2019-blindness-detection)).

### Dataset Breakdown
- **Labeled Training Set (`train.csv`):** Exactly **3,662** verified retinal fundus photographs.
- **Unlabeled Test Set (`test.csv`):** **1,928** retinal photographs (held strictly unlabeled; never used for metric estimation).
- **Resolution:** Variable high-resolution images resized to $300 \times 300 \times 3$ RGB.

### Validated 5-Class Severity Mapping

| Label | Clinical Grade | Morphological Retinal Findings | Action / Clinical Urgency |
|:---:|:---|:---|:---|
| **0** | **No Diabetic Retinopathy** | Normal retina; absence of microaneurysms, hemorrhages, or exudates. | Routine annual eye screening |
| **1** | **Mild NPDR** | Microaneurysms only (focal capillary ballooning). | Rescreen in 6–12 months |
| **2** | **Moderate NPDR** | Microaneurysms, blot hemorrhages, hard exudates, cotton wool spots. | Specialist referral within 3–6 months |
| **3** | **Severe NPDR** | 4-2-1 rule: $\ge 20$ intraretinal hemorrhages in 4 quadrants, venous beading in 2, or IRMA in 1. | Urgent specialist referral (2–4 weeks) |
| **4** | **Proliferative DR (PDR)** | Neovascularization on disc/retina, vitreous hemorrhage, preretinal hemorrhage. | Immediate / urgent surgical intervention |

### Verified Class Distribution & Imbalance
An inspection of the 3,662 training records reveals significant class skew:
- **Grade 0 (No DR):** 1,805 images (49.29%)
- **Grade 1 (Mild):** 370 images (10.10%)
- **Grade 2 (Moderate):** 999 images (27.28%)
- **Grade 3 (Severe):** 193 images (5.27%)
- **Grade 4 (Proliferative):** 295 images (8.06%)

To prevent the model from biasing exclusively toward the majority class (Grade 0), the training pipeline applies **balanced class weighting**:
$$w_c = \frac{N_{\text{total}}}{K \cdot N_c}$$
where $K=5$ classes and $N_c$ is the number of samples in class $c$.

---

## 🧠 3. Deep Learning Architecture & Training Strategy

### Network Architecture
The network combines an ImageNet-pretrained **EfficientNetB3** backbone with a tailored classification head:

```
Input Retinal Image (300 × 300 × 3, RGB, values in [0, 255])
      │
      ▼
EfficientNetB3 Backbone (Pretrained on ImageNet, 391 layers)
      │
      ▼
GlobalAveragePooling2D (GAP)
      │
      ▼
BatchNormalization
      │
      ▼
Dense(256, activation='relu') ──► Dropout(0.4)
      │
      ▼
Dense(128, activation='relu') ──► Dropout(0.3)
      │
      ▼
Dense(5, activation='softmax')
      │
      ▼
Output: [P(Grade 0), P(Grade 1), P(Grade 2), P(Grade 3), P(Grade 4)]
```

### Two-Stage Training Protocol
1. **Stage A — Transfer Learning (Head Only):**
   - The entire convolutional backbone is frozen.
   - Only the custom dense classification layers are trained.
   - **Optimizer:** Adam ($\alpha = 10^{-3}$).
   - **Loss:** Categorical Cross-Entropy with balanced class weights.
2. **Stage B — Fine-Tuning (Top Convolutional Layers):**
   - The top 40 layers of the EfficientNetB3 backbone (including `top_conv` and `block7b` blocks) are unfrozen.
   - Batch normalization layers remain frozen to preserve running statistics.
   - **Optimizer:** Adam with a reduced learning rate ($\alpha = 10^{-5}$) to prevent catastrophic forgetting.
   - **Callbacks:** `ModelCheckpoint` (saving best `val_accuracy`), `EarlyStopping` (patience 4), `ReduceLROnPlateau` (factor 0.5), and `CSVLogger`.

### Correct Image Preprocessing (Avoiding Double-Rescaling)
Keras `EfficientNetB3` includes an internal `Rescaling(scale=1/255.0)` layer. Incompatible normalization (pre-dividing by 255.0 before feeding the tensor) dampens inputs to $[0, 1/255.0]$. VisionCare AI preserves inputs as raw $[0.0, 255.0]$ float tensors:
```python
img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (300, 300), interpolation=cv2.INTER_AREA)
tensor = np.expand_dims(img.astype(np.float32), axis=0) # [0, 255] float
```

---

## 🔍 4. Explainable AI: Grad-CAM Heatmaps

To enable clinical interpretability, VisionCare AI implements **Grad-CAM** (`backend/services/gradcam.py`):
1. Extract feature activation maps $A^k$ from the final convolutional layer (`top_conv`).
2. Calculate the gradient of the predicted class score $y^c$ with respect to each feature map:
   $$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{ij}^k}$$
3. Perform a rectified weighted combination of the feature maps:
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_{k} \alpha_k^c A^k\right)$$
4. Normalize the resulting heatmap to $[0, 1]$, resize to original fundus dimensions, and blend using a JET colormap overlay ($55\%$ original image $+ 45\%$ heatmap).

> **Clinical Interpretation Notice:** Grad-CAM illustrates image features that contributed to the model's numerical activation. It is a transparency tool, not anatomical proof of pathology.

---

## 📊 5. Evaluation Metrics & Benchmark Performance

Model evaluation is performed on a stratified, held-out validation set using `evaluate_model.py`.

### Key Metrics Defined
- **Quadratic Weighted Kappa (QWK):** The international benchmark metric for ordinal diabetic retinopathy grading that penalizes severe cross-grade disagreements exponentially:
  $$\kappa = 1 - \frac{\sum_{i,j} w_{ij} O_{ij}}{\sum_{i,j} w_{ij} E_{ij}}, \quad w_{ij} = \frac{(i - j)^2}{(N - 1)^2}$$
- **Macro and Weighted F1-Scores:** Account for unequal class representation.
- **Multiclass ROC-AUC (One-vs-Rest):** Measures discriminatory capability across all 5 thresholds.

### Benchmark Validation Results (Held-Out Set)
Generated artifacts are saved in `reports/`:
- `reports/metrics.json` – Comprehensive metric summary.
- `reports/classification_report.csv` – Per-class precision, recall, and F1.
- `reports/confusion_matrix.png` – Heatmap of predictions vs. ground truth.
- `reports/training_curves.png` – Loss and accuracy trajectory across epochs.
- `reports/roc_curves.png` – One-vs-Rest ROC curves per grade.

```json
{
  "accuracy": 0.6154,
  "quadratic_weighted_kappa": 0.098,
  "macro_f1": 0.2474,
  "weighted_f1": 0.5121,
  "roc_auc_macro": 0.6781
}
```

---

## 🏗️ 6. Full-Stack Application Architecture

```
Visioncare Ai/
├── backend/
│   ├── app.py                   # Flask server application factory
│   ├── config.py                # JWT, file size, class names & paths
│   ├── database.py              # SQLAlchemy initialization
│   ├── models.py                # User & Prediction database schemas
│   ├── utils.py                 # JWT decorator & DR clinical guidance
│   ├── requirements.txt         # Backend Python dependencies
│   ├── routes/
│   │   ├── auth.py              # /api/register, /api/login, /api/profile
│   │   ├── predict.py           # /api/predict, /api/uploads/<file>, /api/metrics
│   │   └── history.py           # /api/history, /api/report/<id> (PDF)
│   ├── services/
│   │   ├── model_loader.py      # Authentic model loading & inference engine
│   │   ├── gradcam.py           # Grad-CAM heatmap & overlay generation
│   │   └── preprocessing.py    # Retinal image resizing (300×300)
│   ├── saved_model/
│   │   ├── model.keras          # Trained EfficientNetB3 checkpoint
│   │   └── model_config.json    # Architecture metadata
│   ├── uploads/                 # Storage for uploads & Grad-CAM outputs
│   └── reports/                 # Storage for generated PDF reports
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Landing.jsx      # Research overview & features
│   │   │   ├── Login.jsx        # JWT authentication login
│   │   │   ├── Register.jsx     # Clinician account registration
│   │   │   ├── Dashboard.jsx    # Severity breakdown & metric analytics
│   │   │   ├── Predict.jsx      # Retinal image upload & submission
│   │   │   ├── Result.jsx       # Results with Grad-CAM & all 5 class bars
│   │   │   └── History.jsx      # Historical patient records & PDF downloads
│   │   ├── services/api.js      # Axios client with bearer token interceptor
│   │   └── context/AuthContext.jsx # Global user authentication state
├── reports/                     # Model training curves & evaluation metrics
├── tests/
│   └── test_visioncare.py       # Automated unit & integration test suite
├── train.csv                    # 3,662 labeled APTOS 2019 records
├── test.csv                     # 1,928 unlabeled test records
├── train_model.py               # Model training script
└── evaluate_model.py            # Model evaluation script
```

---

## 🚀 7. Installation & Setup Guide (Windows)

### Prerequisites
- **Python 3.10 or 3.11** (TensorFlow 2.15+ support)
- **Node.js 18+ & npm**
- **Git**

### Step 1: Virtual Environment Setup
From the project root directory:
```powershell
# Navigate to backend
cd backend

# Create virtual environment with Python 3.11
python -m venv venv311

# Activate virtual environment (Windows PowerShell)
.\venv311\Scripts\Activate.ps1

# Upgrade pip and install dependencies
pip install -r requirements.txt
cd ..
```

---

### Step 2: Model Training
To train the EfficientNetB3 architecture on the dataset:
```powershell
# Run full training pipeline (10 epochs Stage A + 10 epochs Stage B)
.\backend\venv311\Scripts\python.exe train_model.py --epochs-head 10 --epochs-fine 10 --batch-size 16

# Alternatively, run a verified smoke-test validation run:
.\backend\venv311\Scripts\python.exe train_model.py --smoke-test
```
*The best performing checkpoint will be saved to `backend/saved_model/model.keras`.*

---

### Step 3: Model Evaluation
To evaluate the checkpoint on the held-out validation set and produce evaluation curves:
```powershell
.\backend\venv311\Scripts\python.exe evaluate_model.py
```
*Artifacts (`metrics.json`, `confusion_matrix.png`, `roc_curves.png`) are generated in `reports/`.*

---

### Step 4: Run the Backend Server
```powershell
cd backend
.\venv311\Scripts\python.exe app.py
```
*The Flask API starts at: `http://localhost:5000`.*

---

### Step 5: Run the Frontend Application
In a separate terminal window:
```powershell
cd frontend
npm install
npm run dev
```
*The React UI runs at: `http://localhost:3000`.*

---

## 🧪 8. Automated Testing

Run the automated test suite covering dataset integrity, tensor dimensions, model architecture, Grad-CAM, JWT authentication, prediction endpoints, and PDF generation:
```powershell
.\backend\venv311\Scripts\python.exe -m unittest tests/test_visioncare.py
```
**Test Result:** `9 passed in 10.595s (OK)`.

---

## 🔌 9. REST API Reference

| Method | Endpoint | Auth | Description |
|:---:|:---|:---:|:---|
| `POST` | `/api/register` | Public | Register new clinician/patient account (`name`, `email`, `password`) |
| `POST` | `/api/login` | Public | Authenticate user and issue JWT access token |
| `GET` | `/api/profile` | Bearer Token | Retrieve authenticated user profile and scan count |
| `POST` | `/api/predict` | Bearer Token | Upload retinal image (`multipart/form-data`), run EfficientNetB3 & Grad-CAM |
| `GET` | `/api/history` | Bearer Token | Paginated historical screening records for authenticated user |
| `GET` | `/api/report/<id>`| Bearer Token | Download clinical screening PDF report |
| `GET` | `/api/metrics` | Public | Fetch benchmark validation metrics (`metrics.json`) |
| `GET` | `/api/health` | Public | Server health status, loaded model identity, and compute device |

---

## ⚙️ 10. Environment Variables

| Variable | Default Value | Description |
|:---|:---|:---|
| `SECRET_KEY` | `visioncare-secret-key-dr-2024-secure-32b-min` | Flask session encryption key |
| `JWT_SECRET_KEY`| `visioncare-jwt-secret-dr-2024-secure-32b-min` | Secret key for signing JWT tokens |
| `VISIONCARE_ALLOW_SIMULATION` | `false` | Explicit opt-in simulation flag for development without weights |

---

## ⚠️ 11. Academic Disclaimer & Limitations

1. **Educational & Research Purpose:** This application was developed as a course project for **Deep Learning for Computer Vision**. It is **NOT** a certified medical diagnostic device (CE marked or FDA approved).
2. **Clinical Limitations:** Retinal photography interpretation requires pupil dilation and multi-field assessment. Single-field AI screening outputs must be validated by a licensed ophthalmologist or retinal specialist before determining patient care pathways.
3. **Hardware Fallback:** On native Windows systems running TensorFlow $\ge 2.11$, training and inference execute via multi-threaded CPU instructions unless WSL2 or DirectML is configured.

---

## 📄 License
MIT License — Permitted for academic, educational, and research evaluation.
