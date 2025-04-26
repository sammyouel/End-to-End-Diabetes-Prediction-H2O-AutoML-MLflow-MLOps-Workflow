import os
import sys
import yaml
import pandas as pd
import pandera as pa
from ydata_profiling import ProfileReport
import h2o
from h2o.automl import H2OAutoML
import mlflow
# Import h2o flavor specifically
import mlflow.h2o
import logging
import matplotlib.pyplot as plt # Import for SHAP plot potentially
import traceback # For detailed error logging

# --- Configuration Loading ---
CONFIG_FILE = "config.yaml"

def load_config(config_path):
    """Loads configuration from a YAML file."""
    print(f"Loading configuration from: {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print("Configuration loaded successfully.")
        return config
    except FileNotFoundError:
        print(f"ERROR: Configuration file '{config_path}' not found.")
        sys.exit(1) # Exit if config is missing
    except yaml.YAMLError as e:
        print(f"ERROR: Failed to parse configuration file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: An unexpected error occurred loading configuration: {e}")
        sys.exit(1)

# --- Logging Setup ---
# Setup basic logging to console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Create a logger instance for this script
logger = logging.getLogger(__name__)

# --- Main Execution ---
if __name__ == "__main__":
    logger.info("Starting Enhanced Diabetes Prediction Script...")
    h2o_cluster_running = False # Flag to track H2O cluster status
    run_succeeded = False # Flag to track overall success for MLflow status

    try:
        # Load Configuration
        config = load_config(CONFIG_FILE)

        # --- Create Output Directories ---
        os.makedirs(config['output']['reports_dir'], exist_ok=True)
        os.makedirs(config['output']['model_dir'], exist_ok=True)
        logger.info(f"Ensured output directories exist: '{config['output']['reports_dir']}', '{config['output']['model_dir']}'")

        # --- Check Data File ---
        data_file_path = config['data']['file_path']
        if not os.path.exists(data_file_path):
            logger.error(f"Data file '{data_file_path}' not found. Please place it in the project directory.")
            sys.exit(1)
        logger.info(f"Data file '{data_file_path}' found.")

        # --- 1. Automated EDA (ydata-profiling) ---
        profile_report_path = config['eda']['profile_report_path']
        logger.info("(1) Starting Exploratory Data Analysis (EDA) report generation...")
        try:
            pandas_df_raw = pd.read_csv(data_file_path)
            profile = ProfileReport(pandas_df_raw,
                                    title=f"{config['project_details']['name']} - Data Profiling Report",
                                    explorative=True)
            # Ensure the reports directory exists for the profile report
            os.makedirs(os.path.dirname(profile_report_path), exist_ok=True)
            profile.to_file(profile_report_path)
            logger.info(f"EDA report saved to: {profile_report_path}")
        except Exception as e:
            logger.error(f"Failed to generate EDA report: {e}", exc_info=False) # exc_info=False to avoid duplicate traceback later
            logger.error(traceback.format_exc()) # Manually log traceback if needed

        # --- Data Loading for Validation ---
        logger.info("Loading data with Pandas for validation...")
        pandas_df = pd.read_csv(data_file_path)

        # --- 10. Simple Data Validation (Pandera) ---
        logger.info("(10) Starting Data Validation with Pandera...")
        try:
            schema_dict = {}
            for col, checks_config in config['validation']['schema_checks'].items():
                dtype_str = checks_config.get('dtype', 'object')
                pa_dtype = {
                    'int': pa.Int, 'float': pa.Float, 'str': pa.String, 'bool': pa.Bool, 'datetime': pa.DateTime
                }.get(dtype_str, pa.Object) # Map string to Pandera type

                pa_checks = []
                if 'check_str' in checks_config and checks_config['check_str']:
                    try:
                        check_lambda = eval(checks_config['check_str'])
                        pa_checks.append(pa.Check(check_lambda, element_wise=True, error=f"{col} failed check: {checks_config['check_str']}"))
                    except Exception as eval_e:
                         logger.warning(f"Could not evaluate Pandera check '{checks_config['check_str']}' for column '{col}': {eval_e}. Skipping check.")

                schema_dict[col] = pa.Column(pa_dtype,
                                             checks=pa_checks if pa_checks else None,
                                             nullable=checks_config.get('nullable', False),
                                             required=True)

            schema = pa.DataFrameSchema(schema_dict, strict=True, coerce=True, ordered=False)

            validated_df = schema.validate(pandas_df, lazy=True)
            logger.info("Pandera Data validation successful.")

        except pa.errors.SchemaErrors as err:
            logger.error("Pandera Data Validation Failed!")
            logger.error(f"Schema errors:\n{err.schema_errors}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"An error occurred during data validation: {e}", exc_info=True)
            sys.exit(1)


        # --- 3. MLflow Experiment Tracking Setup ---
        logger.info("(3) Setting up MLflow Experiment Tracking...")
        mlflow.set_experiment(config['mlflow']['experiment_name'])

        # Start MLflow Run using a 'with' block to ensure it always ends
        with mlflow.start_run(run_name=config['mlflow']['run_name']) as run:
            run_id = run.info.run_id
            logger.info(f"MLflow Run started with ID: {run_id}")
            logger.info(f"Track MLflow results: run 'mlflow ui' in terminal (project directory) and navigate to http://localhost:5000")

            # Log configuration parameters
            mlflow.log_params({f"config_{k}": v for k, v in config['data'].items()})
            mlflow.log_params({f"config_{k}": v for k, v in config['h2o'].items()})
            mlflow.log_params({f"config_{k}": v for k, v in config['automl'].items()})
            # Use DIFFERENT keys for overall project details to avoid collision with automl's project_name
            mlflow.log_param("overall_project_name", config['project_details']['name'])
            mlflow.log_param("overall_project_version", config['project_details']['version'])

            # Log the EDA report artifact
            if os.path.exists(profile_report_path):
                 try:
                     mlflow.log_artifact(profile_report_path, artifact_path="eda_report")
                     logger.info("Logged EDA report artifact to MLflow.")
                 except Exception as e:
                     logger.warning(f"Could not log EDA report to MLflow: {e}")

            # --- H2O Initialization ---
            logger.info("Initializing H2O cluster...")
            h2o.init(
                nthreads=config['h2o']['nthreads'],
                max_mem_size=config['h2o']['max_mem_size'],
                port=54321,
            )
            h2o_cluster_running = True
            h2o.cluster().show_status()
            logger.info("H2O cluster initialized successfully.")

            # Load validated data into H2OFrame
            logger.info("Converting validated Pandas DataFrame to H2OFrame...")
            data_h2o = h2o.H2OFrame(validated_df)

            # Set target variable as factor
            target_var = config['data']['target_variable']
            logger.info(f"Setting target variable '{target_var}' as factor...")
            data_h2o[target_var] = data_h2o[target_var].asfactor()

            # Define predictors
            predictors = data_h2o.columns
            predictors.remove(target_var)
            logger.info(f"Predictor variables: {predictors}")
            logger.info(f"Target variable: '{target_var}'")

            # Split Data
            split_ratio = 1.0 - config['data']['test_split_ratio']
            split_seed = config['data']['split_seed']
            logger.info(f"Splitting data (Train={split_ratio*100:.0f}%, Test={(1-split_ratio)*100:.0f}%) with seed {split_seed}...")
            train, test = data_h2o.split_frame(ratios=[split_ratio], seed=split_seed)
            logger.info(f"Training set dimensions: {train.shape}")
            logger.info(f"Testing set dimensions: {test.shape}")
            mlflow.log_param("training_rows", train.shape[0])
            mlflow.log_param("testing_rows", test.shape[0])

            # --- 4. Advanced H2O AutoML Configuration ---
            automl_params = config['automl']
            logger.info(f"(4) Configuring H2O AutoML...")
            logger.info(f"AutoML Parameters: {automl_params}")

            aml = H2OAutoML(
                max_runtime_secs=automl_params['max_runtime_secs'],
                nfolds=automl_params['nfolds'],
                balance_classes=automl_params['balance_classes'],
                sort_metric=automl_params['sort_metric'],
                stopping_metric=automl_params['stopping_metric'],
                stopping_tolerance=automl_params['stopping_tolerance'],
                stopping_rounds=automl_params['stopping_rounds'],
                seed=automl_params['seed'],
                project_name=automl_params['project_name'],
                exclude_algos=automl_params.get('exclude_algos', None),
                include_algos=automl_params.get('include_algos', None)
            )

            # --- Autologging Disabled/Commented Out ---
            # Rely on manual logging

            # Train AutoML model
            logger.info(f"Starting H2O AutoML training (max runtime: {automl_params['max_runtime_secs']} seconds)...")
            aml.train(x=predictors, y=target_var, training_frame=train, leaderboard_frame=test)
            logger.info("AutoML training completed.")

            # --- Results and Evaluation ---
            logger.info("\n--- AutoML Leaderboard ---")
            lb = aml.leaderboard
            lb_df = lb.as_data_frame()
            print(lb_df.to_string())

            # Log leaderboard artifact
            lb_csv_path = os.path.join(config['output']['reports_dir'], "automl_leaderboard.csv")
            lb_df.to_csv(lb_csv_path, index=False)
            mlflow.log_artifact(lb_csv_path, artifact_path="results")
            logger.info(f"Leaderboard saved to {lb_csv_path} and logged to MLflow.")

            # Get Best Model
            best_model = aml.leader
            if not best_model:
                 logger.error("AutoML failed to produce a leader model.")
                 raise RuntimeError("AutoML training did not yield a best model.")

            logger.info(f"\nBest model found: {best_model.model_id} (Algorithm: {best_model.algo})")
            mlflow.set_tag("best_model_id", best_model.model_id)
            mlflow.set_tag("best_model_algo", best_model.algo)

            # Evaluate Best Model on Test Set
            logger.info("Evaluating the best model on the test set...")
            performance = best_model.model_performance(test)
            print("\n--- Best Model Performance on Test Set ---")
            auc = performance.auc()
            logloss = performance.logloss()
            accuracy = performance.accuracy()[0][1]
            f1 = performance.F1()[0][1]
            print(f"AUC: {auc:.4f}")
            print(f"LogLoss: {logloss:.4f}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print("Confusion Matrix:")
            print(performance.confusion_matrix().table)

            # Log performance metrics to MLflow
            logger.info(f"Logging metrics: AUC={auc:.4f}, LogLoss={logloss:.4f}, Accuracy={accuracy:.4f}, F1={f1:.4f}")
            mlflow.log_metric("test_auc", auc)
            mlflow.log_metric("test_logloss", logloss)
            mlflow.log_metric("test_accuracy", accuracy)
            mlflow.log_metric("test_f1", f1)

            # Log confusion matrix artifact
            cm_df = performance.confusion_matrix().table.as_data_frame()
            cm_path = os.path.join(config['output']['reports_dir'], "confusion_matrix.csv")
            cm_df.to_csv(cm_path)
            mlflow.log_artifact(cm_path, artifact_path="results")
            logger.info("Logged confusion matrix to MLflow.")


            # --- Log Model and Register Programmatically ---
            # Define the name for the registered model
            registered_model_name = "PimaDiabetesPredictor-GBM" # Or pull from config if desired
            logger.info(f"Attempting to log and register H2O model as '{registered_model_name}' using mlflow.h2o.log_model...")
            try:
                # Log the H2O model object directly. This should save artifacts (likely MOJO)
                # and register the model or create a new version.
                mlflow.h2o.log_model(
                    h2o_model=best_model,
                    artifact_path="h2o-model", # Path within the run's artifacts
                    # signature=signature, # Optional: Add signature if inferred/defined
                    registered_model_name=registered_model_name
                )
                logger.info(f"Successfully logged and registered model '{registered_model_name}'.")
                run_succeeded = True # Mark success only after model registration attempt
                mlflow.set_tag("status", "Completed")
                mlflow.set_tag("model_registered", "True")

            except Exception as log_model_e:
                logger.error(f"Failed to log/register model using mlflow.h2o.log_model: {log_model_e}", exc_info=True)
                logger.warning("Model artifact might not be properly logged or registered in MLflow.")
                run_succeeded = True # Allow run to complete, but mark model issue
                mlflow.set_tag("status", "Completed_With_LogModel_Error")
                mlflow.set_tag("model_registered", "False")


            # --- 5. Model Explainability (SHAP/VarImp) ---
            # Moved explainability *after* model logging/registration attempt
            if config['explainability']['generate_shap']:
                logger.info("\n(5) Attempting Model Explainability...")
                # (Keep the explainability block as it was in the previous version)
                try:
                    is_shap_supported = hasattr(best_model, 'shap_summary_plot') and \
                                        best_model.algo in ["gbm", "drf", "xgboost"]
                    if is_shap_supported:
                        logger.info("Generating SHAP Summary Plot (requires matplotlib)...")
                        try:
                            shap_plot = best_model.shap_summary_plot(test)
                            fig = plt.gcf()
                            shap_plot_path = os.path.join(config['output']['reports_dir'], "shap_summary_plot.png")
                            fig.savefig(shap_plot_path, bbox_inches='tight')
                            plt.close(fig)
                            mlflow.log_artifact(shap_plot_path, artifact_path="explainability")
                            logger.info(f"SHAP summary plot saved to {shap_plot_path} and logged.")
                        except Exception as plot_err:
                             logger.error(f"Failed to generate or save SHAP plot: {plot_err}", exc_info=True)
                             logger.warning("Falling back to Variable Importance due to SHAP plot error.")
                             is_shap_supported = False

                    if not is_shap_supported:
                        logger.info(f"Generating Variable Importance (SHAP not supported/failed for {best_model.algo})...")
                        varimp = best_model.varimp(use_pandas=True)
                        print("\n--- Variable Importance ---")
                        print(varimp.to_string())
                        varimp_path = os.path.join(config['output']['reports_dir'], "variable_importance.csv")
                        varimp.to_csv(varimp_path, index=False)
                        mlflow.log_artifact(varimp_path, artifact_path="explainability")
                        logger.info(f"Variable importance saved to {varimp_path} and logged.")
                except Exception as explain_e:
                    logger.error(f"Error during explainability step: {explain_e}", exc_info=True)


            # --- 6. Saving Model Locally (Optional - Keep as Backup) ---
            # Commented out the manual saving & logging as we rely on mlflow.h2o.log_model now
            # logger.info("\n(6) Saving the best model locally (backup)...")
            # model_dir = config['output']['model_dir']
            # model_base_name = config['output']['model_base_name']
            # # --- Save H2O Binary format ---
            # binary_model_path_base = os.path.join(model_dir, f"{model_base_name}_binary")
            # try:
            #     os.makedirs(os.path.dirname(binary_model_path_base), exist_ok=True)
            #     binary_path = h2o.save_model(model=best_model, path=binary_model_path_base, force=True)
            #     logger.info(f"Saved H2O Binary model locally to: {binary_path}")
            # except Exception as e:
            #      logger.error(f"Failed to save H2O Binary model locally: {e}", exc_info=True)
            # # --- Export MOJO format ---
            # mojo_model_dir_base = os.path.join(model_dir, f"{model_base_name}_mojo")
            # try:
            #     os.makedirs(mojo_model_dir_base, exist_ok=True)
            #     mojo_path = best_model.download_mojo(path=mojo_model_dir_base, get_genmodel_jar=True)
            #     logger.info(f"Exported MOJO model and JAR locally to directory: {mojo_model_dir_base}")
            # except Exception as e:
            #     logger.error(f"Failed to export MOJO model locally: {e}", exc_info=True)


        # --- End of 'if best_model:' block ---


    # --- End of 'try:' block for main execution ---
    except Exception as e:
        logger.error(f"An critical error occurred during the script execution: {e}", exc_info=True)
        logger.error(traceback.format_exc())
        if 'run_id' in locals() and mlflow.active_run() and mlflow.active_run().info.run_id == run_id:
            if not run_succeeded:
                mlflow.set_tag("status", "Failed")
                try: mlflow.log_param("error_message", str(e)[:250])
                except: pass
        # sys.exit(1) # Commented out

    finally:
        # --- H2O Cluster Shutdown ---
        if h2o_cluster_running:
            logger.info("Shutting down H2O cluster...")
            try:
                h2o.cluster().shutdown(prompt=False)
                logger.info("H2O cluster shut down successfully.")
            except Exception as shutdown_e:
                logger.error(f"Error shutting down H2O cluster: {shutdown_e}", exc_info=True)
        else:
            logger.info("H2O cluster was not running or failed to initialize, skipping shutdown.")

        # Ensure MLflow run ends properly
        if 'run_id' in locals() and mlflow.active_run() and mlflow.active_run().info.run_id == run_id:
             if not run_succeeded and mlflow.active_run().info.status != 'FAILED':
                  mlflow.set_tag("status", "Failed_Incomplete")
             logger.info(f"MLflow Run ({run_id}) completed or ending.")
        elif 'run_id' in locals():
            logger.info(f"MLflow Run ({run_id}) seems to have already ended.")
        else:
             logger.info("No active MLflow run to end in finally block.")


    logger.info("Enhanced Diabetes Prediction Script finished.")