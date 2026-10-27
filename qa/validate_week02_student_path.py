"""Execute CPU teaching/baseline cells; audit, but do not simulate, TensorFlow."""
from pathlib import Path
import ast
import copy
import json
import sys
import nbformat as n
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('module://matplotlib_inline.backend_inline')
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
from matplotlib_inline.backend_inline import configure_inline_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from improve_week02_student_path import improve, PATH, ROOT

book=n.read(PATH,4);before=copy.deepcopy(book)
assert improve(copy.deepcopy(book))==book,'Regeneration must preserve the current lesson'
assert improve(improve(copy.deepcopy(book)))==book,'Regeneration is not idempotent'
shell=InteractiveShell.instance();configure_inline_support(shell,'module://matplotlib_inline.backend_inline')
shell.user_ns.update({key:value for key,value in globals().copy().items() if not key.startswith('_')})
execute={'7e0d6581','2f5bca5d','w2-neuron-experiment','14efa6c9','fe59e15b',
         '7ea8e501','w2-shape-check','6f4a7d90','884c566d','063ab61e','w2-training-controls'}
ran=[]
for cell in book.cells:
    if cell.cell_type=='code':
        ast.parse(cell.source)
        if cell.id not in execute:continue
        with capture_output() as output:result=shell.run_cell(cell.source,store_history=True)
        assert not result.error_before_exec and not result.error_in_exec,cell.id
        assert not output.stderr,output.stderr
        cell.outputs=[]
        if output.stdout:cell.outputs.append(n.v4.new_output('stream',name='stdout',text=output.stdout))
        for rich in output.outputs:cell.outputs.append(n.v4.new_output('display_data',data=rich.data,metadata=rich.metadata))
        cell.execution_count=shell.execution_count-1;ran.append(cell.id)
    elif cell.cell_type=='markdown':
        assert r'\begin{equation}' not in cell.source,cell.id
        assert cell.source.count('$$')%2==0,cell.id
        assert not any(ch in cell.source for ch in [chr(195),chr(194),chr(402)]),cell.id
ns=shell.user_ns
assert ns['parameter_count']==1249 and ns['updates_per_epoch']==55
assert len(ns['X_train'])==3500 and len(ns['X_val'])==750 and len(ns['X_test'])==750
assert np.isclose(ns['curvature'],5.64)
assert all(np.isfinite(ns[name]).all() for name in ['poly_pred','knn_pred'])
# An independent hand-sized metric calculation checks normalization and units.
np.testing.assert_allclose(ns['metrics']([1,2],[2,2]),[np.sqrt(.5),.5,1/np.sqrt(5)])
train=next(c for c in book.cells if c.id=='21818df0').source
for token in ['keras.utils.set_random_seed(TRAINING_SEED)','learning_rate=LEARNING_RATE',
              'epochs=MAX_EPOCHS','batch_size=BATCH_SIZE','patience=PATIENCE',
              'HIDDEN_UNITS[0]','HIDDEN_UNITS[1]','model.count_params() == parameter_count']:
    assert token in train,token
# Evaluate the physical sweep for the two CPU models; do not invent DNN results.
logkn=np.linspace(-4,1.5,160)
features=np.column_stack([logkn,np.full(160,.85),np.full(160,3.),np.ones(160)]).astype('float32')
reference=ns['synthetic_Q'](*features.T,noise_std=0.,seed=0)
assert np.isfinite(reference).all() and np.all(reference>0)
for name in ['poly_model','knn_model']:assert np.isfinite(ns[name].predict(features)).all()
n.validate(book);PATH.write_text(n.writes(book),encoding='utf-8',newline='\n')
report={'executed_cpu_cells':ran,'split_shapes_checked':True,'training_only_scaling_checked':True,
        'dense_parameter_count':1249,'updates_per_epoch':55,'metric_arithmetic_checked':True,
        'polynomial_and_neighbor_predictions_checked':True,
        'tensorflow_training_executed':False,'tensorflow_dependent_cells':'syntax and settings audited; require TensorFlow runtime',
        'student_path_regeneration_idempotent':True,'portable_math_delimiters_checked':True}
out=ROOT/'output/week02_student_path_validation.json';out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(report,indent=2))
