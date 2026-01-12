import joblib
import json 
import pandas as pd
import os
import numpy as np 
from model_training import Dataset
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

parser = argparse.ArgumentParser(description="This scripte uses trained model from model_training.py to make predictions on new data.")
parser.add_argument("--data_folder", type=str, help="Path to the folder containing new data JSON files.",default="Data hit & bounce/per_point_v2")
parser.add_argument("--model_dir", type=str, help="Path to the trained model file.", default="pipeline")
args= parser.parse_args()

model_dir = args.model_dir
data_folder = args.data_folder




def load_pipeline(model_dir):
    """
    Load the trained model pipeline from the specified directory.
    """
    model_event = joblib.load(os.path.join(model_dir, "model_event.joblib"))
    model_bounce_hit = joblib.load(os.path.join(model_dir, "model_bounce_hit.joblib"))
    return model_event , model_bounce_hit

def predict_state( X, model_event , model_bounce_hit):

        # Stage 1 prediction
        if "ball_idx" in X.columns:
            ball_idx_column= X["ball_idx"]  
            X = X.drop(columns=["ball_idx"])
        if "action" in X.columns:
            y= X["action"]
            X = X.drop(columns=["action"])
        event_pred = model_event.predict(X)

        y_pred = np.zeros(len(X), dtype=int)

        # For detected events, classify bounce vs hit
        idx_event = np.where(event_pred == 1)[0]
        if len(idx_event) > 0:
            sub_pred = model_bounce_hit.predict(X.iloc[idx_event])
            y_pred[idx_event] = np.where(sub_pred == 1, 2, 1)

        #save the prediction with the input data as a csv file

        output_df = X.copy()
        output_df= output_df.drop(columns=['Vx', 'Vy', 'Ax',
       'Ay', 'lag_1_Vx', 'lag_1_Vy', 'lag_1_Ax', 'lag_1_Ay', 'lag_2_Vx',
       'lag_2_Vy', 'lag_2_Ax', 'lag_2_Ay','lead_1_Vx','lead_1_Vy',
       'lead_1_Ax','lead_1_Ay','lead_2_Vx','lead_2_Vy','lead_2_Ax','lead_2_Ay'])
        try:
            output_df["ball_idx"] = ball_idx_column
            output_df["action"] = y
            output_df["action"] = output_df["action"]
        except:
            print("Attention: ball_idx or action column not found in the input data")
            print("Saving without these columns")
            pass
        output_df["pred_action"] = y_pred
        output_df["pred_action"]= output_df["pred_action"].map({0:"air",1:"bounce",2:"hit"})
        
        output_df.to_csv("model_predictions.csv", index=False)    
        return output_df

def csv_to_ball_jsons(df, output_folder: str):
    """
    Convert a CSV file into per-ball JSON files.

    Parameters
    ----------
    csv_path : str
        Path to the input CSV file.
    output_folder : str
        Folder where JSON files will be stored.
    """

    
    

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Group by ball index
    for ball_idx, ball_df in df.groupby("ball_idx"):
        ball_data = {}

        for _, row in ball_df.iterrows():
            frame = str(int(row["frame"]))

            ball_data[frame] = {
                "x": None if pd.isna(row["X"]) else row["X"],
                "y": None if pd.isna(row["Y"]) else row["Y"],
                "visible": bool(row["visible"]),
                "action": row["action"],
                "pred_action": row["pred_action"],
            }

        # Write JSON file
        output_path = os.path.join(
            output_folder, f"ball_data_{ball_idx}.json"
        )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(ball_data, f, indent=2)


### unsupervide model 
def build_features(df):
    df = df.copy()

    # Velocity deltas
    df["dVx"] = df["Vx"] - df["lag_1_Vx"]
    df["dVy"] = df["Vy"] - df["lag_1_Vy"]

    # Absolute values (spike detection)
    df["abs_Ax"] = df["Ax"].abs()
    df["abs_Ay"] = df["Ay"].abs()
    df["abs_dVx"] = df["dVx"].abs()
    df["abs_dVy"] = df["dVy"].abs()

    # Vy sign flip → bounce indicator
    df["vy_sign_flip"] = ((df["Vy"] * df["lag_1_Vy"]) < 0).astype(int)

    return df


def detect_events(df, percentile=95):
    df = df.copy()

    df["event_score"] = (
        df["abs_Ax"]
        + df["abs_Ay"]
        + df["abs_dVx"]
        + df["abs_dVy"]
    )

    threshold = np.percentile(df["event_score"], percentile)
    df["is_event"] = df["event_score"] >= threshold

    return df

def cluster_events(df, n_clusters=3):
    feature_cols = [
        "abs_Ax",
        "abs_Ay",
        "abs_dVx",
        "abs_dVy",
        "vy_sign_flip",
    ]

    event_df = df[df["is_event"]].copy()

    X = event_df[feature_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type="full",
        random_state=42,
    )

    event_df["cluster"] = gmm.fit_predict(X_scaled)

    return event_df, gmm, scaler
   

def assign_pred_action(df, event_df, cluster_map):
    df = df.copy()
    df["pred_action"] = "air"

    event_df = event_df.copy()
    event_df["pred_action"] = event_df["cluster"].map(cluster_map)

    df.loc[event_df.index, "pred_action"] = event_df["pred_action"]

    return df
def unsupervised_action_detection(df):
    df_feat = build_features(df)
    df_evt = detect_events(df_feat, percentile=90)

    event_df, gmm, scaler = cluster_events(df_evt, n_clusters=3)
  
    stats = cluster_stats(event_df)
    cluster_action_map = map_pred_action_by_sign_flip(stats.to_dict("records"))
    event_df["pred_action"] = event_df["cluster"].map(cluster_action_map)
    df["pred_action"] =  event_df["pred_action"]

    return df, cluster_action_map

def unsupervised_action_detection(df):
    df_feat = build_features(df)
    df_evt = detect_events(df_feat, percentile=90)

    event_df, gmm, scaler = cluster_events(df_evt, n_clusters=3)

    stats_df = cluster_stats(event_df)
    cluster_action_map = map_pred_action(stats_df)

    event_df["pred_action"] = event_df["cluster"].map(cluster_action_map)

    df_evt["pred_action"] = "air"
    df_evt.loc[event_df.index, "pred_action"] = event_df["pred_action"]

    return df_evt, cluster_action_map


def map_pred_action(cluster_stats_df):
    cluster_stats_df = cluster_stats_df.copy()

    # Bounce = max sign flip rate
    bounce_cluster = cluster_stats_df.sort_values(
        "sign_flip_rate", ascending=False
    ).iloc[0]["cluster"]

    # Hit = max impulse
    cluster_stats_df["impulse"] = (
        cluster_stats_df["mean_abs_Ax"] + cluster_stats_df["mean_abs_Ay"]
    )
    hit_cluster = cluster_stats_df.sort_values(
        "impulse", ascending=False
    ).iloc[0]["cluster"]

    mapping = {}
    for cid in cluster_stats_df["cluster"]:
        if cid == bounce_cluster:
            mapping[cid] = "bounce"
        elif cid == hit_cluster:
            mapping[cid] = "hit"
        else:
            mapping[cid] = "air"

    return mapping

def cluster_stats(event_df):
    stats = []

    for cid in sorted(event_df["cluster"].unique()):
        c = event_df[event_df["cluster"] == cid]

        stats.append({
            "cluster": cid,
            "mean_abs_Ax": c["abs_Ax"].mean(),
            "std_abs_Ax": c["abs_Ax"].std(),
            "mean_abs_Ay": c["abs_Ay"].mean(),
            "std_abs_Ay": c["abs_Ay"].std(),
            "sign_flip_rate": c["vy_sign_flip"].mean(),
            "count": len(c)
        })

    return pd.DataFrame(stats)

def map_pred_action_by_sign_flip(cluster_stats_list):
    # Sort clusters by sign_flip_rate
    sorted_stats = sorted(cluster_stats_list, key=lambda x: int(x["sign_flip_rate"]))
    # Assign actions
    mapping = {}
    if len(sorted_stats) == 3:
        mapping[sorted_stats[0]["cluster"]] = "air"
        mapping[sorted_stats[1]["cluster"]] = "bounce"
        mapping[sorted_stats[2]["cluster"]] = "hit"
    elif len(sorted_stats) == 2:
        mapping[sorted_stats[0]["cluster"]] = "air"
        mapping[sorted_stats[1]["cluster"]] = "hit"
    else:
        mapping[sorted_stats[0]["cluster"]] = "air"
    return mapping





def compare_action_pred(data: pd.DataFrame):
    """
    Compare action and pred_action columns.

    Returns:
    - accuracy (float)
    - confusion matrix (DataFrame)
    - per-class accuracy (DataFrame)
    """

    # Accuracy globale
    accuracy = (data["action"] == data["pred_action"]).mean()

    # Matrice de confusion
    confusion = pd.crosstab(
        data["action"],
        data["pred_action"],
        rownames=["action"],
        colnames=["pred_action"],
        dropna=False
    )

    # Accuracy par classe
    per_class_accuracy = (
        data.assign(correct=data["action"] == data["pred_action"])
            .groupby("action")["correct"]
            .mean()
            .reset_index(name="accuracy")
    )

    return accuracy, confusion, per_class_accuracy




model_event , model_bounce_hit = load_pipeline(model_dir)
print("The models are loaded ")
data= Dataset(data_folder)
print("Data is being prepared")
data = data.prepare()
print( "the data is ready")
data_unsupervised,cluster_mapping= unsupervised_action_detection(data)




Model_prediction = predict_state(data, model_event , model_bounce_hit)
print("The prediction is done !")

csv_to_ball_jsons(data_unsupervised,"prediction_unsupervised")
csv_to_ball_jsons(Model_prediction,"prediction_supervised")
print ("========================================================")
print("The prediction Json files are saved in <prediction_supervised> and <prediction_unsupervised> ")
print ("========================================================")
acc, conf_mat, class_acc = compare_action_pred(Model_prediction)

print("Accuracy globale:", acc)
print("\nMatrice de confusion:\n", conf_mat)
print("\nAccuracy par classe:\n", class_acc)