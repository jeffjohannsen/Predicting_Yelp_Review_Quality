
import numpy as np
import pandas as pd

from model_pipeline import ModelDetailsStorage, run_full_pipeline

if __name__ == "__main__":
    model_details = ModelDetailsStorage()
    for data in ['text', 'non_text', 'both']:
        for target in ['T2_CLS_ufc_>0', 'T5_CLS_ufc_level_TD']:
            for model in ['Log Reg', 'Forest Cls', 'HGB Cls', 'XGB Cls']:
                run_full_pipeline(use_cv=True, print_results=True,
                                  save_results=True, question='td',
                                  records=10000, data=data,
                                  target=target, model=model,
                                  scalar='power', balancer='smote')
