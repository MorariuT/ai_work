"""Sample task: simple regression model.

The contestant cloudpickles a callable that maps a numpy array of features
(``[N, 1]``) to a numpy array of predictions (``[N]``). The scorer sends the
test features once via ``exchange`` and forwards the predictions to the
judge as ``submission.csv``.
"""

import io
from pathlib import Path

import numpy as np
import pandas as pd
import zipfile
import cloudpickle
from predictor import exchange
test_path = "test_data" if Path("test_data").is_file() else "test_data.zip"

with zipfile.ZipFile(test_path, "r") as z:
    test_a = pd.read_csv(z.open("state_values.csv"))
    test_b = pd.read_csv(z.open("state_actions.csv"))
    test_c = pd.read_csv(z.open("policy_actions.csv"))

packet = {
    "subtask1": test_a.to_dict("records"),
    "subtask2": test_b.to_dict("records"),
    "subtask3": test_c.to_dict("records"),
}

features = packet 
# send 
buf = cloudpickle.dumps(packet)
# received 
response = exchange(buf)

# handle 
predictions = cloudpickle.loads(response)

# make submission 

# make submission 
data = {
    "subtask1": test_a.to_dict("records"),
    "subtask2": test_b.to_dict("records"),
    "subtask3": test_c.to_dict("records"),

}
rows = []
for name,s_id in zip(("subtask1","subtask2","subtask3"), (1,2,3)):
    
    current = data[name]
    if name in predictions:
        # handle it
        subtask_prediction = predictions[name]
        
        for response in subtask_prediction:
            
            rows.append(
                {
                    "subtaskID": s_id,
                    "datapointID": response['datapointID'],
                    "answer": response['answer'],
                }
            )

    # nu sunt sigur daca am nevoie de asta
    else:
        for row in current:
            rows.append({
                "subtaskID": s_id,
                "datapointID": row['id'],
                "answer": -100, 
            })

pd.DataFrame(
    rows
).to_csv("submission.csv", index=False)
