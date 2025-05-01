import streamlit as st
import os
import tempfile
import shutil
import pandas as pd
from datetime import datetime
from eval.batch_eval import batch_evaluate_files 
from eval.compute_global_score_from_batches import aggregate_batch_metrics
import numpy as np

st.set_page_config(
    page_title="PINKCC Challenge 2025",
    page_icon="icon.png"
)

st.title("Welcome to the PINKCC 2025 Data Challenge Evaluation Test Platform")
st.write(
    """
    This platform allows you to upload your data for evaluation.
    We will use this exact platform for the final evaluation, so this allows you to check that your data is in the correct format !
    """
)

# --- Team Name Field as dropdown from CSV ---
team_names_df = pd.read_csv("web/team_names.csv")
team_names_list = team_names_df["Team Name"].dropna().astype(str).tolist()
team_name = st.selectbox("Team Name (required before evaluation)", [""] + team_names_list)

st.write("### Upload Evaluation Labels Folder")
evaluation_files = st.file_uploader(
    "Upload all files from the evaluation labels folder.",
    type=None,
    accept_multiple_files=True,
    key="evaluation"
)

st.write("### Upload Prediction Labels Folder")
prediction_files = st.file_uploader(
    "Upload all files from the prediction labels folder.",
    type=None,
    accept_multiple_files=True,
    key="prediction"
)

def get_file_names(files):
    return sorted([file.name for file in files]) if files else []

def save_files_to_tempdir(files, temp_dir):
    paths = []
    for file in files:
        path = os.path.join(temp_dir, file.name)
        with open(path, "wb") as f:
            f.write(file.getbuffer())
        paths.append(path)
    return paths

# --- Store results in session_state to persist after download ---
if "batch_metrics" not in st.session_state:
    st.session_state["batch_metrics"] = None
if "table_df" not in st.session_state:
    st.session_state["table_df"] = None
if "global_metrics" not in st.session_state:
    st.session_state["global_metrics"] = None

if evaluation_files and prediction_files:
    st.write("### File Matching Check")
    evaluation_file_names = {file.name for file in evaluation_files}
    prediction_file_names = {file.name for file in prediction_files}

    missing_in_predictions = evaluation_file_names - prediction_file_names
    missing_in_evaluations = prediction_file_names - evaluation_file_names

    if not missing_in_predictions and not missing_in_evaluations:
        st.success("All files match between the two folders!")

        files_key = (
            tuple(get_file_names(evaluation_files)),
            tuple(get_file_names(prediction_files))
        )
        if (
            "eval_paths" not in st.session_state or
            "pred_paths" not in st.session_state or
            st.session_state.get("files_key") != files_key
        ):
            # Clean up old temp dirs if they exist
            if "eval_temp_dir" in st.session_state:
                shutil.rmtree(st.session_state["eval_temp_dir"], ignore_errors=True)
            if "pred_temp_dir" in st.session_state:
                shutil.rmtree(st.session_state["pred_temp_dir"], ignore_errors=True)

            eval_temp_dir = tempfile.mkdtemp()
            pred_temp_dir = tempfile.mkdtemp()
            eval_paths = save_files_to_tempdir(evaluation_files, eval_temp_dir)
            pred_paths = save_files_to_tempdir(prediction_files, pred_temp_dir)
            st.session_state["eval_paths"] = eval_paths
            st.session_state["pred_paths"] = pred_paths
            st.session_state["files_key"] = files_key
            st.session_state["eval_temp_dir"] = eval_temp_dir
            st.session_state["pred_temp_dir"] = pred_temp_dir
        else:
            eval_paths = st.session_state["eval_paths"]
            pred_paths = st.session_state["pred_paths"]

        num_classes = st.number_input("Number of classes", min_value=3, max_value=5, value=3)
        batch_size = st.number_input("Batch size", min_value=1, max_value=20, value=1)
        # Disable evaluation button if team name is empty

        col1, _ = st.columns([2, 6])
        with col1:
            eval_button = st.button("Run Evaluation", disabled=(team_name.strip() == ""))
            progress_bar = st.empty()

        if eval_button:
            batch_metrics = []
            total_batches = (len(eval_paths) + batch_size - 1) // batch_size

            columns = [
                "Batch", "P_1", "P_2", "P_mean", "R_1", "R_2", "R_mean", "F2_1", "F2_2", "F2_mean",
                "DDS", "Batch Size", "Prediction Files", "Evaluation Files"
            ]

            table_df = pd.DataFrame({
                "Batch": [f"{i+1}/{total_batches}" for i in range(total_batches)],
                "P_1": [np.nan] * total_batches,
                "P_2": [np.nan] * total_batches,
                "P_mean": [np.nan] * total_batches,
                "R_1": [np.nan] * total_batches,
                "R_2": [np.nan] * total_batches,
                "R_mean": [np.nan] * total_batches,
                "F2_1": [np.nan] * total_batches,
                "F2_2": [np.nan] * total_batches,
                "F2_mean": [np.nan] * total_batches,
                "DDS": [np.nan] * total_batches,
                "Batch Size": [np.nan] * total_batches,
                "Prediction Files": [""] * total_batches,
                "Evaluation Files": [""] * total_batches,
            })

            st.write("### Batch Results")
            table_placeholder = st.empty()

            with st.spinner("Running batch evaluation..."):
                for i, (batch_result, pred_batch, eval_batch) in enumerate(
                    batch_evaluate_files(
                        pred_paths,
                        eval_paths,
                        num_classes=num_classes,
                        batch_size=batch_size,
                    )
                ):
                    table_df.at[i, "P_1"] = round(float(batch_result["P_1"]), 4)
                    table_df.at[i, "P_2"] = round(float(batch_result["P_2"]), 4)
                    table_df.at[i, "P_mean"] = round(float(batch_result["P_mean"]), 4)
                    table_df.at[i, "R_1"] = round(float(batch_result["R_1"]), 4)
                    table_df.at[i, "R_2"] = round(float(batch_result["R_2"]), 4)
                    table_df.at[i, "R_mean"] = round(float(batch_result["R_mean"]), 4)
                    table_df.at[i, "F2_1"] = round(float(batch_result["F_beta_1"]), 4)
                    table_df.at[i, "F2_2"] = round(float(batch_result["F_beta_2"]), 4)
                    table_df.at[i, "F2_mean"] = round(float(batch_result["F_beta_mean"]), 4)
                    table_df.at[i, "DDS"] = round(float(batch_result["dilated_dice_score"]), 4)
                    table_df.at[i, "Batch Size"] = batch_result["batch_size"]
                    table_df.at[i, "Prediction Files"] = ", ".join([os.path.basename(f) for f in pred_batch])
                    table_df.at[i, "Evaluation Files"] = ", ".join([os.path.basename(f) for f in eval_batch])
                    batch_metrics.append(batch_result)
                    table_placeholder.dataframe(
                        table_df,
                        use_container_width=True,
                        hide_index=True
                    )
                    progress_bar.progress((i + 1) / total_batches)
            st.success("All batches processed!")

            st.session_state["batch_metrics"] = batch_metrics
            st.session_state["table_df"] = table_df
            global_metrics = aggregate_batch_metrics(batch_metrics)
            st.session_state["global_metrics"] = global_metrics

            # Show aggregated metrics as a table with team name as first column
            st.write("### Aggregated Metrics")
            # Convert all tensor or numpy values to float and round to 3 decimals
            cleaned_metrics = {}
            for k, v in global_metrics.items():
                # Convert torch.Tensor or np.generic to float and round
                if hasattr(v, "item"):
                    cleaned_metrics[k] = round(float(v.item()), 3)
                else:
                    try:
                        cleaned_metrics[k] = round(float(v), 3)
                    except Exception:
                        cleaned_metrics[k] = v

            aggregated_df = pd.DataFrame([{
                "Team Name": team_name,
                "Date": datetime.now().strftime("%d/%m/%Y"),
                **cleaned_metrics
            }])

            # Reorder columns to have Team Name first
            cols = aggregated_df.columns.tolist()
            if "Team Name" in cols:
                cols.insert(0, cols.pop(cols.index("Team Name")))
            aggregated_df = aggregated_df[cols]

            st.dataframe(
                aggregated_df,
                use_container_width=True,
                hide_index=True
            )

            # Add a warning with a smiley
            st.warning("⚠️ The aggregated F2 score is not the average of the F2 scores, it is a global score computed over all volumes at once.")


    else: # If there are missing files
        if missing_in_predictions:
            st.error("The following files are missing in the prediction labels folder:")
            for file_name in missing_in_predictions:
                st.write(file_name)
        if missing_in_evaluations:
            st.error("The following files are missing in the evaluation labels folder:")
            for file_name in missing_in_evaluations:
                st.write(file_name)
        st.warning("Please rerun the page and upload the correct files.")

