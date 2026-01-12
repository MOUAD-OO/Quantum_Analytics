import os 
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier






class Dataset :
    def __init__(self,data_directory):
        self.data_directory=data_directory

    @staticmethod
    def get_ball_index(filename):
        name=filename.split("_")
        ball_index = name[-1].replace(".json", "")
        
        return int(ball_index)


    def load_json(self):
    
        self.raw_data =[]
        for filename in os.listdir(self.data_directory):
            if filename.endswith(".json"):
                with open(os.path.join(self.data_directory, filename), "r") as f:
                    self.raw_data_line=json.load(f)
                    self.raw_data_line["ball_idx"]=self.get_ball_index(filename)
                    self.raw_data.append(self.raw_data_line)
        return self.raw_data



    @staticmethod
    def interpolate_ball_trajectories(data):
        """
        Interpolate X,Y per ball_idx, ignoring leading gaps.
        Assumes df is sorted by ['ball_idx', 'frame'].
        """

        df = data.copy()

        def interpolate_one_ball(group):
            # Work only on X, Y
            xy = group[['X', 'Y']]

            # Find first valid index (both X and Y present)
            first_valid = xy.dropna().index.min()

            if first_valid is None:
                # Entire ball has no valid data → do nothing
                return group

            # Interpolate only AFTER first valid observation
            group.loc[first_valid:, ['X', 'Y']] = (
                group.loc[first_valid:, ['X', 'Y']]
                .interpolate(method='linear', limit_direction='forward')
            )

            return group

        df = (
            df
            .groupby('ball_idx', group_keys=False)
            .apply(interpolate_one_ball)
        )

        return df


    @staticmethod
    def velocity(data):

        prev_x = None 
        prev_y = None 
        prev_frame = None
        vx= None
        vy= None
        
        for i in range(1, len(data)):
            if data.loc[i,"ball_idx"] != data.loc[i-1,"ball_idx"]:
                prev_x = None 
                prev_y = None 
                prev_frame = None
                data.loc[i,"Vx"] = np.nan
                data.loc[i,"Vy"] = np.nan
                continue
            x = data.loc[i,"X"]
            y = data.loc[i,"Y"]
            frame = data.loc[i,"frame"]
            if data.loc[i,"visible"] == False:
                data.loc[i,"Vx"] = vx
                data.loc[i,"Vy"] = vy

                continue
            if prev_frame is not None:
                dt = frame - prev_frame
                vx = (x - prev_x) / dt
                vy = (y - prev_y) / dt
                data.loc[i,"Vx"] = vx
                data.loc[i,"Vy"] = vy
            prev_x = x
            prev_y = y
            prev_frame = frame
        return data

    @staticmethod
    def acceleration(data):

        prev_vx = None 
        prev_vy = None 
        prev_frame = None
        ax= None
        ay= None
        
        for i in range(1, len(data)):
            if data.loc[i,"ball_idx"] != data.loc[i-1,"ball_idx"]:
                prev_vx = None 
                prev_vy = None 
                prev_frame = None
                data.loc[i,"Ax"] = np.nan
                data.loc[i,"Ay"] = np.nan
                continue
            vx = data.loc[i,"Vx"]
            vy = data.loc[i,"Vy"]
            frame = data.loc[i,"frame"]
            if prev_vx is None:
                data.loc[i,"Ax"] = 0
                data.loc[i,"Ay"] = 0
                prev_vx = vx
                prev_vy = vy
                prev_frame = frame
                continue
            if prev_frame is not None:
                dt = frame - prev_frame
                ax = (vx - prev_vx) / dt
                ay = (vy - prev_vy) / dt
                data.loc[i,"Ax"] = ax
                data.loc[i,"Ay"] = ay
            prev_vx = vx
            prev_vy = vy
            prev_frame = frame
        return data



    def json_to_dataframe(self):
        rows = []

        for json_line in self.raw_data:
            ball_idx = json_line["ball_idx"]

            for frame, value in json_line.items():
                if frame == "ball_idx":
                    continue

                rows.append({
                    "ball_idx": ball_idx,
                    "frame": int(frame),
                    "X": value["x"],
                    "Y": value["y"],
                    "visible": value["visible"],
                    "action": value["action"]
                })

        df = pd.DataFrame(rows)

        # Ensure correct ordering
        df = df.sort_values(["ball_idx", "frame"]).reset_index(drop=True)
        
        df = self.interpolate_ball_trajectories(df)
        df = self.velocity(df)
        df = self.acceleration(df)

        df.to_csv("DataFrame.csv", index=False)
        return df
    
    def feature_engineering(self):
        data=self.dataframe
        data['visible'] = data['visible'].replace(
        {True:1, False:0, 'True':1, 'False':0, 'true':1, 'false':0}).infer_objects()
        #The ball movement is indipendent of the ball index
        #so we can use the 80% of the data for training and 20% for testing
        data = data.sort_values(["ball_idx", "frame"]).reset_index(drop=True)
        num_cols = ["X", "Y", "Vx", "Vy", "Ax", "Ay"]

        data[num_cols] = data[num_cols].apply(pd.to_numeric, errors="coerce")

        lags = [1, 2]

        hist_features = ["Vx", "Vy", "Ax", "Ay"]

        for lag in lags:
            for feat in hist_features:
                data[f"lag_{lag}_{feat}"] = (data.groupby("ball_idx")[feat].shift(lag))
        # Forward (future)
        for lag in lags:
            for feat in hist_features:
                data[f"lead_{lag}_{feat}"] = (data.groupby("ball_idx")[feat].shift(-lag))

        # Replace non-existing values with 0
        lag_lead_cols = [
            c for c in data.columns
            if c.startswith("lag_") or c.startswith("lead_")
        ]

        data[lag_lead_cols] = data[lag_lead_cols].fillna(0.0)
        data = data.dropna()
        return data 

    def prepare(self):        
        self.raw_data = self.load_json()
        self.dataframe=self.json_to_dataframe()
        self.dataframe=self.feature_engineering()
        return self.dataframe







class Model:
    def __init__(self,dataframe: pd.DataFrame):
        self.df=dataframe
        self.model_event=XGBClassifier(n_estimators=300,max_depth=5,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,
                                        objective="binary:logistic",tree_method="hist",eval_metric="aucpr")
        
        self.model_bounce_hit =XGBClassifier(n_estimators=300,max_depth=5,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,
                                        objective="binary:logistic",tree_method="hist",eval_metric="aucpr")

    def feature_engineering(self):
        data= self.df
        data['ball_idx'] = data['ball_idx'].astype(int) 
        mask = data["action"] != "air"
        data_filtered = data.loc[mask]
        #========================================================
        #data for the first model 
        #========================================================
        self.data_training=data.where(data['ball_idx']<=320).dropna().reset_index(drop=True) 
        self.data_testing=data.where(data['ball_idx']>320).dropna().reset_index(drop=True)

        #remove the ball_idx column from the training data 
        self.data_training=self.data_training.drop(columns=["ball_idx"])
        self.data_testing=self.data_testing.drop(columns=["ball_idx"])
        
        
        self.X_training =self.data_training.drop(columns=["action"])
        self.X_testing =self.data_testing.drop(columns=["action"])

        self.target_training = self.data_training["action"].map({"air":0,"bounce":1,"hit":1})
        self.target_testing = self.data_testing["action"].map({"air":0,"bounce":1,"hit":2})
        
     
        #========================================================
        #data for the Second model 
        #========================================================
        #========================================================
        self.data_training_2=data_filtered.where(data_filtered['ball_idx']<=320).dropna().reset_index(drop=True) 
        self.data_testing_2=data_filtered.where(data_filtered['ball_idx']>320).dropna().reset_index(drop=True)

        #remove the ball_idx column from the training data 
        self.data_training_2=self.data_training_2.drop(columns=["ball_idx"])
        self.data_testing_2=self.data_testing_2.drop(columns=["ball_idx"])
        
        
        self.X_training_2 =self.data_training_2.drop(columns=["action"])
        self.X_testing_2 =self.data_testing_2.drop(columns=["action"])

        self.target_training_2 = self.data_training_2["action"].map({"bounce":0,"hit":1})
        self.target_testing_2 = self.data_testing_2["action"].map({"bounce":0,"hit":1})
        

     




    def train_model(self):
        weight=[1,50,50]
        class_weights = {
            0:1,
            1: 2,
           
        }
        


        sample_weights = compute_sample_weight(class_weight=class_weights,y=self.target_training)

        self.model_event.fit(self.X_training, self.target_training,sample_weight= sample_weights)
        self.model_bounce_hit.fit(self.X_training_2,self.target_training_2)


    def predict_state(self, X):
        # Stage 1 prediction
        if "ball_idx" in X.columns:
            ball_idx_column= X["ball_idx"]  
            X = X.drop(columns=["ball_idx"])
        if "action" in X.columns:
            y= X["action"]
            X = X.drop(columns=["action"])
        event_pred = self.model_event.predict(X)

        y_pred = np.zeros(len(X), dtype=int)

        # For detected events, classify bounce vs hit
        idx_event = np.where(event_pred == 1)[0]
        if len(idx_event) > 0:
            sub_pred = self.model_bounce_hit.predict(X.iloc[idx_event])
            y_pred[idx_event] = np.where(sub_pred == 1, 2, 1)

        #save the prediction with the input data as a csv file

        output_df = X.copy()
        try:
            output_df["ball_idx"] = ball_idx_column
            output_df["action"] = y
        except:
            print("Attention: ball_idx or action column not found in the input data")
            print("Saving without these columns")
            pass
        output_df["pred_action"] = y_pred
        output_df.to_csv("model_predictions.csv", index=False)    
        return y_pred
        

    def evaluate_model(self):
       
        print(classification_report(
        self.target_testing,
        self.predict_state(self.X_testing),
        digits=4
        ))

    def save_pipline(self):
  
        pipeline_dir = "pipeline"
        if not os.path.exists(pipeline_dir):
            os.makedirs(pipeline_dir)

        # 2. Save trained models
        joblib.dump(
            self.model_event,
            os.path.join(pipeline_dir, "model_event.joblib")
        )

        joblib.dump(
            self.model_bounce_hit,
            os.path.join(pipeline_dir, "model_bounce_hit.joblib")
        )

        # 3. Save model hyperparameters 
        params = {
            "model_event_params": self.model_event.get_params(),
            "model_bounce_hit_params": self.model_bounce_hit.get_params()
        }

        with open(os.path.join(pipeline_dir, "model_parameters.json"), "w") as f:
            json.dump(params, f, indent=4)

        print("Models and parameters successfully saved in 'pipeline/'")
        with open(os.path.join(pipeline_dir, "model_metrics.txt"),'w') as f :
            f.write(classification_report(
                    self.target_testing,
                    self.predict_state(self.X_testing),
                    digits=4
                    ))


def main():
    DF= Dataset("Data hit & bounce/per_point_v2")
    dataframe =DF.prepare()
    dataframe.to_csv("prepared_dataframe.csv", index=False)
    print("Dataframe prepared for model training:")
    model=Model(dataframe)
    model.feature_engineering()
    model.train_model()
    model.evaluate_model()
    model.save_pipline()

if __name__ == "__main__":
    main()
    




        
        