import io
import cloudpickle
import numpy as np
import pandas as pd
class Submission:
    """
    Acesta este modelul pe care-l vei folosi ca si submisie.
    """

    def __init__(self):
        # Incarcati Q-tablelul/modelul
        # self.model = ... 

        pass 
    def __call__(self, data: bytes) -> bytes:
        """
        Functia __call__ este obligatorie.
        """
        test_data = cloudpickle.loads(data)
        """
        test_data contine urmatoarele key-uri:
        test_data.keys() = dict(["subtask1","subtask2","subtask3"])
        """

        """ 
        Mai in jos se afla un DEMO cum sa faci o predictie
        """
        prediction = {}
        subtask1 = []
        subtask2 = []
        subtask3 = []
        # handle subtask 1
        subtask1_data = test_data["subtask1"]
        for row in subtask1_data:
            """ 
            row = {'id': *, 'signal_power': *, 'tide_code': *, 'dolphin_boost': *}
            """
            id = row["id"]
            # De aici urmeaza implementarea ta.

            

            subtask1.append(
                {
                    "datapointID": id,
                    "answer": -100,
                }
            )
        subtask2_data = test_data["subtask2"]
        for row in subtask2_data:
            """ 
            row = {'id': *, 'signal_power': *, 'tide_code': *, 'dolphin_boost': *, 'action': *}
            """
            id = row["id"]
            # De aici urmeaza implementarea ta.
            subtask2.append(
                {
                    "datapointID": id,
                    "answer": -100,
                }
            )
            
        subtask3_data = test_data["subtask3"]
        for row in subtask3_data:
            """ 
            row = {'id': *, 'signal_power': *, 'tide_code': *, 'dolphin_boost': *, 'question': *}
            """
            id = row["id"]
            # De aici urmeaza implementarea ta.
            subtask3.append(
                {
                    "datapointID": id,
                    "answer": -100,
                }
            )


        """ 
        Asigura-te ca formatul la "predictions" este: 
        predictions: {
        "subtask1": ...,
        "subtask2":...,
        "subtask3":...
        },unde subtask* = {
        "datapointID":[...],
        "answer":[...]
        } pentru fiecare ID primit din test_data["subtask*"]["datapointID"]
        !!! Nerespectarea formatului va duce la eroare (;
        """
        prediction = {
            "subtask1": subtask1,
            "subtask2": subtask2,
            "subtask3": subtask3
        }
        prediction = cloudpickle.dumps(prediction)
        return prediction

print("Writing your model...")
with open("submission.pkl", "wb") as f:
    cloudpickle.dump(Submission(), f)
print("Wrote submission.pkl")

print("Loading your model...")
with open("submission.pkl", "rb") as f:
    model = cloudpickle.load(f)
print("Model loaded")
random_data = {
    "subtask1": [
        {
            "id": 157,
            "signal_power": 0,
            "tide_code": 0,
            "dolphin_boost": 0
        }
    ],
    "subtask2": [],
    "subtask3": []
}
bytes_data = cloudpickle.dumps(random_data)
returned_bytes = model(bytes_data)
loaded_response = cloudpickle.loads(returned_bytes)
print(f"response: {loaded_response}")