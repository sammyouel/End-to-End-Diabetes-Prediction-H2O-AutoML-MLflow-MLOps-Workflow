# app.py
# Basic Flask API to serve predictions using a saved H2O MOJO model.

import os
import sys
import glob
import pandas as pd
from flask import Flask, request, jsonify
import h2o
import logging
import yaml # For loading config to find model dir

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - API - %(message)s')
logger = logging.getLogger(__name__)

# --- Load Config to Find Model Path ---
CONFIG_FILE = "config.yaml"
try:
    with open(CONFIG_FILE, 'r') as f:
        config = yaml.safe_load(f)
    MODEL_DIR = config['output']['model_dir']
    MODEL_BASE_NAME = config['output']['model_base_name']
    # Construct expected MOJO directory/file name pattern
    mojo_pattern = os.path.join(MODEL_DIR, f"{MODEL_BASE_NAME}_mojo", "*.zip")
    logger.info(f"Looking for MOJO model using pattern: {mojo_pattern}")
except Exception as config_e:
     logger.error(f"CRITICAL ERROR: Could not load configuration from {CONFIG_FILE} to find model path: {config_e}", exc_info=True)
     sys.exit(1)


# --- Find MOJO Model and Jar ---
try:
    mojo_zip_files = glob.glob(mojo_pattern)
    if not mojo_zip_files:
        # Fallback: Check directly in MODEL_DIR if subfolder structure wasn't created as expected by H2O download_mojo
        mojo_pattern_alt = os.path.join(MODEL_DIR, f"{MODEL_BASE_NAME}_mojo.zip") # Assuming direct file name
        mojo_zip_files = glob.glob(mojo_pattern_alt)
        if not mojo_zip_files:
             raise FileNotFoundError(f"No MOJO .zip file found matching pattern '{mojo_pattern}' or '{mojo_pattern_alt}'")

    MOJO_PATH = mojo_zip_files[0] # Assume first match is the one
    logger.info(f"Identified MOJO model path: {MOJO_PATH}")

    # Find h2o-genmodel.jar (essential for mojo_predict_pandas)
    mojo_dir = os.path.dirname(MOJO_PATH)
    genmodel_jar_path = os.path.join(mojo_dir, "h2o-genmodel.jar")
    if not os.path.exists(genmodel_jar_path):
         # Check one level up if MOJO was in a subdirectory relative to the jar download path
         genmodel_jar_path_alt = os.path.join(os.path.dirname(mojo_dir), "h2o-genmodel.jar")
         if os.path.exists(genmodel_jar_path_alt):
              genmodel_jar_path = genmodel_jar_path_alt
              logger.info(f"Found h2o-genmodel.jar one level up: {genmodel_jar_path}")
         else:
              # If still not found, H2O might find it via classpath, but warn user.
              logger.warning(f"h2o-genmodel.jar not found directly near MOJO ({genmodel_jar_path} or {genmodel_jar_path_alt}). Ensure it's in Java classpath for predictions.")
              # Set to None or keep path? H2O might handle it, let's keep path for potential explicit use.
    else:
         logger.info(f"Found h2o-genmodel.jar at: {genmodel_jar_path}")


except FileNotFoundError as e:
    logger.error(f"CRITICAL ERROR: {e}")
    logger.error("Ensure 'predict_diabetes.py' has run successfully and generated the MOJO model in the configured 'models' directory.")
    sys.exit(1)
except Exception as e:
    logger.error(f"CRITICAL ERROR during model path configuration: {e}", exc_info=True)
    sys.exit(1)

# --- API Endpoints ---
@app.route('/')
def home():
    """Basic home route."""
    return jsonify({"message": "H2O Diabetes Prediction API. Use POST /predict."})

@app.route('/predict', methods=['POST'])
def predict():
    """Receives patient data JSON, returns prediction using MOJO."""
    logger.info("Received request on /predict endpoint.")

    if not request.is_json:
        logger.warning("Request content type is not JSON.")
        return jsonify({"error": "Request must be JSON"}), 400

    try:
        data = request.get_json()
        logger.info(f"Received input data: {data}")

        # --- Basic Input Validation ---
        # These MUST match the columns the model was trained on, in the correct order if model requires it
        required_features = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                             "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
        missing = [feature for feature in required_features if feature not in data]
        if missing:
            logger.warning(f"Missing required features in input: {missing}")
            return jsonify({"error": f"Missing required features: {missing}"}), 400

        # Convert JSON to DataFrame matching required feature order
        input_df = pd.DataFrame([data], columns=required_features)
        logger.debug(f"Input data converted to DataFrame:\n{input_df.to_string()}")

        # --- Prediction using H2O MOJO ---
        logger.info(f"Performing prediction using MOJO: {MOJO_PATH}")
        # Pass genmodel_jar_path explicitly if found, otherwise let H2O try to find it
        predictions_df = h2o.mojo_predict_pandas(
            dataframe=input_df,
            mojo_path=MOJO_PATH,
            genmodel_jar_path=genmodel_jar_path if os.path.exists(genmodel_jar_path) else None,
            verbose=False
        )
        logger.info("Prediction successful.")
        logger.debug(f"Prediction output DataFrame:\n{predictions_df.to_string()}")

        # --- Format Response ---
        output = predictions_df.to_dict('records')[0]
        response = {
            "predicted_outcome": int(output.get('predict', -1)),
            "probability_class_0": output.get('p0', None),
            "probability_class_1": output.get('p1', None)
        }
        logger.info(f"Sending response: {response}")
        return jsonify(response)

    except FileNotFoundError as fnf_err:
         logger.error(f"MOJO/jar likely not found during prediction: {fnf_err}", exc_info=True)
         return jsonify({"error": "Model files missing, API cannot predict."}), 500
    except ValueError as ve:
        logger.error(f"Value error (check data types/format): {ve}", exc_info=True)
        return jsonify({"error": f"Data format error: {ve}"}), 400
    except Exception as e:
        logger.error(f"Unexpected prediction error: {e}", exc_info=True)
        return jsonify({"error": "Internal server error during prediction."}), 500

# --- Run Flask App ---
if __name__ == '__main__':
    logger.info(f"Attempting to start Flask server on http://0.0.0.0:5000")
    # Use host='0.0.0.0' for accessibility from outside container/local network
    app.run(host='0.0.0.0', port=5000, debug=False) # debug=False is recommended for stability