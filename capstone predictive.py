#!/usr/bin/env python
# coding: utf-8

# #  Train using Azure Machine Learning Compute
# 
# * Initialize a Workspace
# * Create an Experiment
# * Introduction to AmlCompute
# * Submit an AmlCompute script run using a persistent compute target
# * Download the fitted model from the run output artifacts

# ## Prerequisites
# If you are using an Azure Machine Learning Compute Instance, **Experiment** is a logical container in an Azure ML Workspace. It hosts run records which can include run metrics and output artifacts from your experiments. Please ensure `azureml-core` is installed on the machine running Jupyter.

# In[ ]:


# Check core SDK version number
import azureml.core

print("SDK version:", azureml.core.VERSION)


# ## Initialize a Workspace
# 
# Initialize a workspace object from the previous experiment

# In[ ]:


from azureml.core import Workspace

# The workspace information from the previous experiment has been pre-filled for you.
subscription_id = "d539000f-b3f2-4f7e-a1ca-ad8bf072973b"
resource_group = "ADHeimer-rg"
workspace_name = "Heimer2"

ws = Workspace(subscription_id=subscription_id, resource_group=resource_group, workspace_name=workspace_name)
print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep = '\n')


# ## Create An Experiment
# 
# **Experiment** is a logical container in an Azure ML Workspace. It hosts run records which can include run metrics and output artifacts from your experiments.

# In[ ]:


from azureml.core import Experiment

# The experiment name has been pre-filled for you.
experiment_name = "peppe344"
experiment = Experiment(workspace = ws, name = experiment_name)


# ## Introduction to AmlCompute
# 
# Azure Machine Learning Compute is managed compute infrastructure that allows the user to easily create single to multi-node compute of the appropriate VM Family. It is created **within your workspace region** and is a resource that can be used by other users in your workspace. It autoscales by default to the max_nodes, when a job is submitted, and executes in a containerized environment packaging the dependencies as specified by the user. 
# 
# Since it is managed compute, job scheduling and cluster management are handled internally by Azure Machine Learning service. 
# 
# For more information on Azure Machine Learning Compute, please read [this article](https://docs.microsoft.com/azure/machine-learning/service/how-to-set-up-training-targets#amlcompute)
# 
# **Note**: As with other Azure services, there are limits on certain resources (for eg. AmlCompute quota) associated with the Azure Machine Learning service. Please read [this article](https://docs.microsoft.com/azure/machine-learning/service/how-to-manage-quotas) on the default limits and how to request more quota.
# 
# 
# The training script is already created for you. Let's have a look.

# ### Create project directory
# 
# Create a directory that will contain all the necessary code from your local machine that you will need access to on the remote resource. This includes the training script, and any additional files your training script depends on

# In[ ]:


import os
import shutil

project_folder = os.path.join(".", experiment_name)
os.makedirs(project_folder, exist_ok=True)
shutil.copy('script.py', project_folder)


# ### Create environment
# 
# Create Docker based environment with scikit-learn installed.

# In[ ]:


import hashlib
from azureml.core import Environment
from azureml.core.runconfig import DockerConfiguration
from azureml.core.conda_dependencies import CondaDependencies

myenv = Environment.get(ws, 'AzureML-AutoML', '120')

# Enable Docker
docker_config = DockerConfiguration(use_docker=True)


# ### Provision as a persistent compute target (Basic)
# 
# You can provision a persistent AmlCompute resource by simply defining two parameters thanks to smart defaults. By default it autoscales from 0 nodes and provisions dedicated VMs to run your job in a container. This is useful when you want to continously re-use the same target, debug it between jobs or simply share the resource with other users of your workspace.
# 
# * `vm_size`: VM family of the nodes provisioned by AmlCompute. Simply choose from the supported_vmsizes() above
# * `max_nodes`: Maximum nodes to autoscale to while running a job on AmlCompute

# In[ ]:


from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

# Choose a name for your CPU cluster
cluster_name = "pepp4"

# Verify that cluster does not exist already
try:
    cluster = ComputeTarget(workspace=ws, name=cluster_name)
    print('Found existing cluster, use it.')
except ComputeTargetException:
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_DS12_V2',
                                                           max_nodes=4)
    cluster = ComputeTarget.create(ws, cluster_name, compute_config)

cluster.wait_for_completion(show_output=True)


# ### Configure & Run

# In[ ]:


import uuid
from azureml.core import ScriptRunConfig
from azureml._restclient.models import RunTypeV2
from azureml._restclient.models.create_run_dto import CreateRunDto
from azureml._restclient.run_client import RunClient

codegen_runid = str(uuid.uuid4())
client = RunClient(experiment.workspace.service_context, experiment.name, codegen_runid, experiment_id=experiment.id)

# To test with new training / validation datasets, replace the default dataset id(s) taken from parent run below
training_dataset_id = 'acba9a7e-d58f-4309-aca5-ee5ecc499dcd'
dataset_arguments = ['--training_dataset_id', training_dataset_id]

create_run_dto = CreateRunDto(run_id=codegen_runid,
                              parent_run_id='AutoML_855cee12-8bca-402e-9d8b-963f77212cb3_46',
                              description='AutoML Codegen Script Run',
                              target=cluster_name,
                              run_type_v2=RunTypeV2(
                                  orchestrator='Execution', traits=['automl-codegen']))
src = ScriptRunConfig(source_directory=project_folder, 
                      script='script.py', 
                      arguments=dataset_arguments, 
                      compute_target=cluster, 
                      environment=myenv,
                      docker_runtime_config=docker_config)
run_dto = client.create_run(run_id=codegen_runid, create_run_dto=create_run_dto)
 
run = experiment.submit(config=src, run_id=codegen_runid)
run


# Note: if you need to cancel a run, you can follow [these instructions](https://aka.ms/aml-docs-cancel-run).

# In[ ]:


get_ipython().run_cell_magic('time', '', '# Shows output of the run on stdout.\nrun.wait_for_completion(show_output=True)')


# In[ ]:


run.get_metrics()


# ### Download Fitted Model

# In[ ]:


import joblib

# Load the fitted model from the script run.

# Note that if training dependencies are not installed on the machine
# this notebook is being run from, this step can fail.
try:
    run.download_file("outputs/model.pkl", "model.pkl")
    model = joblib.load("model.pkl")
except ImportError:
    print('Required dependencies are missing; please run pip install azureml-automl-runtime.')
    raise


# You can now inference using this model.  
# For classification/regression, call `model.predict()`  
# For forecasting, call `model.forecast()`
