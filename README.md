![mlflow](https://s20.directupload.net/images/220301/qlrp88lw.png)

# Getting started with mlflow

This repo aims to show some first steps with mlflow.
* Tracking
* Models
* Model Registry

## Contents

- [Getting started with mlflow](#getting-started-with-mlflow)
  * [General](#general)
  * [Simulation on localhost](#simulation-on-localhost)
  * [Further reading](#further-reading)
    

## General

To use mlflow one general needs:
* a server on which mlflow runs (incl. the ui)
* an artifact store
* a database as well as a connector (e.g. sqlite)

Note that a database is not mandatory for tracking. If not specified, mlflow will create a specific folder structure on the disk instead. 
However, using the Model Registry is not possible in that case.

#### Pyspark Serving

Note that the pyspark serving notebook is optional.  
If you want to use it, you need to install pyspark and pyarrow as defined in the requirements.  
Note that a corresponding java version needs to be installed as well to run spark.  

## Simulation on localhost

Here, localhost simulates a cloud on which mlflow is running. A dedicated folder resp. database simulates the artifact store and remote database. 

First, set up a virtual environment given the requirements. 

Then, create an empty database, e.g. via sqlite which should be built in for macOs.  

``` console
cd cloud_mock
sqlite 3
```

``` console
.save mlflow.db
.exit
```

Then start mlflow ui in your active virtual environment and start mlflow server while you're working directory is *cloud_mock*.

```console
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./../cloud_mock/artifacts \
    --host 127.0.0.1
```

Note that one reaches the minimal setup via

```console
mlflow ui
```

but this has some disadvantages as described above. 

### Create an endpoint

Once a model is registered, one can serve the model 

```console
mlflow models serve -m "models:/{model_name}/{model_version}" -p yourport
```

Make sure to set the tracking uri in the corresponding terminal.

```console
export MLFLOW_TRACKING_URI='http://localhost:5000'
```

Note that there are few other opportunities, e.g. building a docker-image or building specific images
to deploy the model to different cloud platforms.

## Further reading

* [Official documentation](https://www.mlflow.org/docs/latest/index.html)
* [Managed MLflow by databricks](https://databricks.com/de/product/managed-mlflow) 
* [Mlflow docker as a oneliner](https://github.com/Toumash/mlflow-docker)
* [Databricks pricing](https://databricks.com/product/pricing)
* [GCP Setup proposal](https://medium.com/@Sushil_Kumar/setting-up-mlflow-on-google-cloud-for-remote-tracking-of-machine-learning-experiments-b48e0122de04)
* [AWS Setup proposal](https://aws.amazon.com/blogs/machine-learning/managing-your-machine-learning-lifecycle-with-mlflow-and-amazon-sagemaker/
)

