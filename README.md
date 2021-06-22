# mlflow - Getting started

This repo aims to show some first steps with mlflow: 
* Tracking
* MLFLow Models
* Model Registry

## Contents

- [mlflow - Getting started](#mlflow---getting-started)
  * [General](#general)
  * [Simulation on localhost](#simulation-on-localhost)
  * [Further reading](#further-reading)
    

## General

To use MLFlow one general needs:
* a server on which mlflow runs (incl. the ui)
* an artifact store
* a database as well as a connector (e.g. sqlite)

Note that a database is not mandatory for tracking. If not specified, mlflow will create a specific folder strucure on the disk instead. 
However, using the Model Registry is not possible in that case. 

## Simulation on localhost

Here, localhost simulates a cloud on which mlflow is running. A dedicated folder resp. database simulates the artifact store and remote database. 

First, set up a virtual environment given the requirements and create an empty database, e.g. via sqlite.  
Then start mlflow ui in your active virtual environment and navigate to folder "my_cloud".

```console
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./../my_cloud/artifacts \
    --host 127.0.0.1
```

Note that one reaches the minimal setup via

```console
mlflow ui
```

but this has some disadvantages as described above. 

### Create an endpoint

Once a model is registred, one can create an endpoint via conda typing the following 

```console
mlflow models serve -m "models:/{model_name}/{model_version} -p yourport
```

Make sure to set the tracking uri in the corresponding terminal.

```console
export MLFLOW_TRACKING_URI='http://localhost:5000'
```

## Further reading

* [Official documentation](https://www.mlflow.org/docs/latest/index.html)
* [Managed MLflow by databricks](https://databricks.com/de/product/managed-mlflow) 
* [Mlflow docker as a oneliner](https://github.com/Toumash/mlflow-docker)
* [GCP Setup proposal](https://medium.com/@Sushil_Kumar/setting-up-mlflow-on-google-cloud-for-remote-tracking-of-machine-learning-experiments-b48e0122de04)
* [AWS Setup proposal](https://aws.amazon.com/blogs/machine-learning/managing-your-machine-learning-lifecycle-with-mlflow-and-amazon-sagemaker/
)

