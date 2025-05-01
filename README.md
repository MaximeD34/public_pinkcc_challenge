# Welcome to the 2025 Pinkcc Challenge Evaluation Repository

Here you can inspect how we will evaluate your model, and evaluate your own validation set yourself.

## 📊 Best Way to Evaluate

The easiest and most reliable way to evaluate your model is by using the **same Streamlit app** we will use during the official evaluation.

### 🔧 Setup Instructions

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```
Install dependencies:
```bash
pip install -r requirements.txt
```
Run the app:
```bash
python -m streamlit run main_web.py
```
From there, just drag and drop your evaluation and prediction files into the app. It will validate your submission format and notify you if anything is wrong.

⚠️ If your predictions are not valid in the app, they won't be valid during our official evaluation either.

🐍 Prefer Using Python Directly?

You can also use the Python API directly by calling the eval_pipeline function in eval_pipeline.py.

Note: This requires passing stacked PyTorch tensors as input. In the official evaluation, you will be asked to submit .nii.gz files whose filenames must match ours exactly. If you're not sure, use the streamlit app.
