# """
# submit_pipeline.py
# ------------------
# Submits the EDT training pipeline to Azure ML.
# """

# import os
# import sys
# import time
# import argparse
# from pathlib import Path

# from azure.identity import ClientSecretCredential
# from azure.ai.ml import MLClient, command, Input, Output, dsl
# from azure.ai.ml.entities import Environment, AmlCompute, Model, Data
# from azure.ai.ml.constants import AssetTypes
# from azure.core.exceptions import AzureError, ResourceNotFoundError


# PROJECT_ROOT = Path(__file__).resolve().parent.parent

# # ── ML Settings ──────────────────────────────────────────────────────
# CLUSTER_NAME  = "edt-cpu-cluster"
# CLUSTER_SIZE  = "Standard_DS3_v2"
# ENV_NAME      = "edt-training-env"
# DATA_ASSET    = "japan-pizza-delivery-data"
# DATA_VERSION  = "1"
# LOCAL_CSV     = str(PROJECT_ROOT / "japan_pizza_delivery.csv")
# MODEL_NAME    = "edt-model-japan"
# EXPERIMENT    = "edt-model-training"
# MAX_RETRIES   = 3
# RETRY_DELAY_S = 5


# # ── Logging ──────────────────────────────────────────────────────────
# def log(msg):
#     print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# # ── Retry helper ─────────────────────────────────────────────────────
# def with_retry(fn, label):
#     for attempt in range(1, MAX_RETRIES + 1):
#         try:
#             return fn()
#         except AzureError as e:
#             if attempt == MAX_RETRIES:
#                 log(f"X {label} failed: {e}")
#                 raise
#             log(f"! {label} retry {attempt}/{MAX_RETRIES}")
#             time.sleep(RETRY_DELAY_S)


# # ── Azure ML Client ───────────────────────────────────────────────────
# def get_ml_client():
#     # ✅ FIX: read ALL env vars inside function
#     # never at module level
#     tenant_id       = os.environ.get("AZURE_TENANT_ID", "")
#     client_id       = os.environ.get("AZURE_CLIENT_ID", "")
#     client_secret   = os.environ.get("AZURE_CLIENT_SECRET", "")
#     subscription_id = os.environ.get("AZURE_SUBSCRIPTION_ID", "")
#     resource_group  = os.environ.get("AZURE_RESOURCE_GROUP", "rg-tshele-5296")
#     workspace_name  = os.environ.get("AZURE_ML_WORKSPACE",   "edt-ml-workspace")

#     # Debug — print lengths to confirm secrets are loaded
#     print(f"tenant_id length:       {len(tenant_id)}")
#     print(f"client_id length:       {len(client_id)}")
#     print(f"client_secret length:   {len(client_secret)}")
#     print(f"subscription_id length: {len(subscription_id)}")
#     print(f"resource_group:         {resource_group}")
#     print(f"workspace_name:         {workspace_name}")

#     # Validate before using
#     if not tenant_id:
#         raise ValueError("AZURE_TENANT_ID is empty")
#     if not client_id:
#         raise ValueError("AZURE_CLIENT_ID is empty")
#     if not client_secret:
#         raise ValueError("AZURE_CLIENT_SECRET is empty")
#     if not subscription_id:
#         raise ValueError("AZURE_SUBSCRIPTION_ID is empty")

#     credential = ClientSecretCredential(
#         tenant_id     = tenant_id,
#         client_id     = client_id,
#         client_secret = client_secret,
#     )
#     client = MLClient(
#         credential          = credential,
#         subscription_id     = subscription_id,
#         resource_group_name = resource_group,
#         workspace_name      = workspace_name,
#     )
#     log(f"Connected: workspace={client.workspace_name} RG={client.resource_group_name}")
#     return client


# # ── Dataset upload ────────────────────────────────────────────────────
# def ensure_data_asset(client):
#     csv_path = Path(LOCAL_CSV)

#     if not csv_path.exists():
#         log(f"CSV not found: {csv_path}")
#         raise FileNotFoundError(csv_path)

#     try:
#         existing = client.data.get(name=DATA_ASSET, version=DATA_VERSION)
#         log(f"Dataset already registered: {existing.name}:{existing.version}")
#         return f"azureml:{DATA_ASSET}:{DATA_VERSION}"
#     except ResourceNotFoundError:
#         pass

#     log(f"Uploading dataset from {csv_path}...")
#     data = Data(
#         name        = DATA_ASSET,
#         version     = DATA_VERSION,
#         path        = str(csv_path),
#         type        = AssetTypes.URI_FILE,
#         description = "Japan pizza delivery dataset",
#     )
#     registered = with_retry(
#         lambda: client.data.create_or_update(data),
#         label="Upload dataset",
#     )
#     log(f"Dataset uploaded: {registered.name}:{registered.version}")
#     return f"azureml:{registered.name}:{registered.version}"


# # ── Compute cluster ───────────────────────────────────────────────────
# def ensure_compute(client):
#     try:
#         client.compute.get(CLUSTER_NAME)
#         log(f"Compute exists: {CLUSTER_NAME}")
#     except ResourceNotFoundError:
#         log(f"Creating compute: {CLUSTER_NAME}...")
#         compute = AmlCompute(
#             name          = CLUSTER_NAME,
#             size          = CLUSTER_SIZE,
#             min_instances = 0,
#             max_instances = 4,
#         )
#         with_retry(
#             lambda: client.compute.begin_create_or_update(compute).result(),
#             label="Create compute",
#         )
#         log(f"Compute created: {CLUSTER_NAME}")
#     return CLUSTER_NAME


# # ── Environment ───────────────────────────────────────────────────────
# def ensure_environment(client):
#     env_yml = PROJECT_ROOT / "environment.yml"
#     if not env_yml.exists():
#         raise FileNotFoundError(f"environment.yml not found: {env_yml}")

#     env = Environment(
#         name       = ENV_NAME,
#         image      = "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
#         conda_file = str(env_yml),
#     )
#     registered = with_retry(
#         lambda: client.environments.create_or_update(env),
#         label="Create environment",
#     )
#     log(f"Environment ready: {registered.name}:{registered.version}")
#     return f"{registered.name}:{registered.version}"


# # ── Pipeline ──────────────────────────────────────────────────────────
# def build_pipeline(data_path, compute, env):
#     preprocess = command(
#         name         = "preprocess",
#         display_name = "Step 1 - Preprocess",
#         code         = "./src",
#         command      = (
#             "python preprocess.py "
#             "--input ${{inputs.raw_data}} "
#             "--output ${{outputs.processed}}"
#         ),
#         inputs  = {"raw_data":   Input(type=AssetTypes.URI_FILE)},
#         outputs = {"processed":  Output(type=AssetTypes.URI_FOLDER, mode="rw_mount")},
#         environment = env,
#         compute     = compute,
#     )

#     train = command(
#         name         = "train",
#         display_name = "Step 2 - Train",
#         code         = "./src",
#         command      = (
#             "python train.py "
#             "--data ${{inputs.processed_data}} "
#             "--output ${{outputs.model}}"
#         ),
#         inputs  = {"processed_data": Input(type=AssetTypes.URI_FOLDER)},
#         outputs = {"model":          Output(type=AssetTypes.URI_FOLDER, mode="rw_mount")},
#         environment = env,
#         compute     = compute,
#     )

#     @dsl.pipeline(
#         name        = "edt-training-pipeline",
#         description = "EDT model training pipeline",
#         tags        = {"market": "japan"},
#     )
#     def pipeline(raw_data):
#         prep_step  = preprocess(raw_data=raw_data)
#         train_step = train(processed_data=prep_step.outputs.processed)
#         return {"model": train_step.outputs.model}

#     return pipeline(raw_data=Input(type=AssetTypes.URI_FILE, path=data_path))


# # ── Model registration ────────────────────────────────────────────────
# def register_model_after_job(client, job_name):
#     log("Registering model...")
#     model = Model(
#         name        = MODEL_NAME,
#         path        = f"azureml://jobs/{job_name}/outputs/model",
#         type        = AssetTypes.CUSTOM_MODEL,
#         description = f"EDT model from job {job_name}",
#         tags        = {"market": "japan", "framework": "scikit-learn"},
#     )
#     registered = with_retry(
#         lambda: client.models.create_or_update(model),
#         label="Register model",
#     )
#     log(f"Model registered: {registered.name}:{registered.version}")


# # ── Args ──────────────────────────────────────────────────────────────
# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--experiment", default=EXPERIMENT)
#     parser.add_argument("--no_wait",    action="store_true")
#     return parser.parse_args()


# # ── Main ──────────────────────────────────────────────────────────────
# def main():
#     args = parse_args()

#     log("Connecting to Azure ML...")
#     client = get_ml_client()

#     log("Checking dataset...")
#     data_path = ensure_data_asset(client)

#     log("Checking compute...")
#     compute = ensure_compute(client)

#     log("Checking environment...")
#     env = ensure_environment(client)

#     log("Building pipeline...")
#     pipeline_job = build_pipeline(data_path, compute, env)

#     log("Submitting job...")
#     job = with_retry(
#         lambda: client.jobs.create_or_update(
#             pipeline_job,
#             experiment_name=args.experiment,
#         ),
#         label="Submit pipeline",
#     )
#     log(f"Job submitted: {job.name}")
#     log(f"Studio URL   : {job.studio_url}")

#     if args.no_wait:
#         log("--no_wait flag set. Exiting.")
#         return

#     log("Streaming logs (Ctrl+C to detach)...")
#     try:
#         client.jobs.stream(job.name)
#     except KeyboardInterrupt:
#         log("Detached. Job is still running in Azure ML.")
#         return

#     final = client.jobs.get(job.name)
#     log(f"Pipeline status: {final.status}")

#     if final.status == "Completed":
#         register_model_after_job(client, job.name)
#         log("Pipeline completed successfully.")
#     else:
#         log(f"Pipeline failed: {final.status}")
#         log(f"Check logs: {final.studio_url}")
#         sys.exit(1)


# if __name__ == "__main__":
#     main()
"""
submit_pipeline.py
------------------
Submits the EDT training pipeline to Azure ML.

Authentication:
  - GitHub Actions : WorkloadIdentityCredential (federated OIDC — no client secret)
  - Local dev      : DefaultAzureCredential fallback (az login / env vars)
"""

import os
import sys
import time
import argparse
from pathlib import Path

from azure.identity import WorkloadIdentityCredential, DefaultAzureCredential
from azure.ai.ml import MLClient, command, Input, Output, dsl
from azure.ai.ml.entities import Environment, AmlCompute, Model, Data
from azure.ai.ml.constants import AssetTypes
from azure.core.exceptions import AzureError, ResourceNotFoundError

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── ML Settings ──────────────────────────────────────────────────────
CLUSTER_NAME = "edt-cpu-cluster"
CLUSTER_SIZE = "Standard_DS3_v2"
ENV_NAME = "edt-training-env"
DATA_ASSET = "japan-pizza-delivery-data"
DATA_VERSION = os.environ.get("DATA_VERSION", "1")
LOCAL_CSV = str(PROJECT_ROOT / "japan_pizza_delivery.csv")
MODEL_NAME = "edt-model-japan"
EXPERIMENT = "edt-model-training"
MAX_RETRIES = 3
RETRY_DELAY_S = 5


# ── Logging ──────────────────────────────────────────────────────────
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def log_section(title):
    print(f"\n{'─' * 50}", flush=True)
    print(f"  {title}", flush=True)
    print(f"{'─' * 50}", flush=True)


# ── Retry helper ─────────────────────────────────────────────────────
def with_retry(fn, label):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn()
        except AzureError as e:
            if attempt == MAX_RETRIES:
                log(f"X {label} failed after {MAX_RETRIES} attempts: {e}")
                raise
            log(f"! {label} attempt {attempt}/{MAX_RETRIES} failed — retrying in {RETRY_DELAY_S}s...")
            time.sleep(RETRY_DELAY_S)


# ── Credential ────────────────────────────────────────────────────────
def get_credential():
    """
    In GitHub Actions, azure/login@v2 sets AZURE_CLIENT_ID, AZURE_TENANT_ID,
    and AZURE_FEDERATED_TOKEN_FILE automatically when using federated credentials.
    WorkloadIdentityCredential reads those env vars directly — no client secret needed.

    Locally, falls back to DefaultAzureCredential (picks up az login or env vars).
    """
    client_id = os.environ.get("AZURE_CLIENT_ID")
    tenant_id = os.environ.get("AZURE_TENANT_ID")
    token_file = os.environ.get("AZURE_FEDERATED_TOKEN_FILE")

    if client_id and tenant_id and token_file:
        log("Auth: WorkloadIdentityCredential (federated OIDC)")
        return WorkloadIdentityCredential(
            client_id=client_id,
            tenant_id=tenant_id,
        )

    log("Auth: DefaultAzureCredential (local fallback)")
    return DefaultAzureCredential()


# ── Azure ML Client ───────────────────────────────────────────────────
def get_ml_client():
    log_section("Azure ML Client")

    required = {
        "AZURE_SUBSCRIPTION_ID": os.environ.get("AZURE_SUBSCRIPTION_ID"),
        "AZURE_RESOURCE_GROUP": os.environ.get("AZURE_RESOURCE_GROUP"),
        "AZURE_ML_WORKSPACE": os.environ.get("AZURE_ML_WORKSPACE"),
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        log("X Missing required environment variables:")
        for var in missing:
            log(f"    - {var} is {'empty' if var in os.environ else 'not set'}")
        log("")
        log("  Fix: add these as GitHub repository secrets:")
        log("  Settings -> Secrets and variables -> Actions -> New repository secret")
        raise ValueError(f"Missing env vars: {', '.join(missing)}")

    subscription_id = required["AZURE_SUBSCRIPTION_ID"]
    resource_group = required["AZURE_RESOURCE_GROUP"]
    workspace_name = required["AZURE_ML_WORKSPACE"]

    log(f"Subscription  : {subscription_id}")
    log(f"Resource group: {resource_group}")
    log(f"Workspace     : {workspace_name}")

    try:
        credential = get_credential()
        client = MLClient(
            credential=credential,
            subscription_id=subscription_id,
            resource_group_name=resource_group,
            workspace_name=workspace_name,
        )
        log(f"Connected: workspace={client.workspace_name} RG={client.resource_group_name}")
        return client
    except Exception as e:
        log(f"X Failed to connect to Azure ML: {e}")
        raise


# ── Dataset upload ────────────────────────────────────────────────────
def ensure_data_asset(client):
    log_section("Data Asset")

    csv_path = Path(LOCAL_CSV)
    if not csv_path.exists():
        log(f"X CSV not found at: {csv_path}")
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    log(f"CSV found: {csv_path}")

    try:
        existing = client.data.get(name=DATA_ASSET, version=DATA_VERSION)
        log(f"Dataset already registered: {existing.name}:{existing.version}")
        return f"azureml:{DATA_ASSET}:{DATA_VERSION}"
    except ResourceNotFoundError:
        log("Dataset not found — uploading...")

    data = Data(
        name=str(DATA_ASSET),
        version=str(DATA_VERSION),
        path=str(csv_path),
        type=AssetTypes.URI_FILE,
        description="Japan pizza delivery dataset",
    )
    registered = with_retry(
        lambda: client.data.create_or_update(data),
        label="Upload dataset"
    )
    log(f"Dataset uploaded: {registered.name}:{registered.version}")
    return f"azureml:{registered.name}:{registered.version}"


# ── Compute cluster ───────────────────────────────────────────────────
def ensure_compute(client):
    log_section("Compute Cluster")

    try:
        cluster = client.compute.get(CLUSTER_NAME)
        log(f"Compute exists: {cluster.name} ({cluster.size})")
        return CLUSTER_NAME
    except ResourceNotFoundError:
        log(f"Cluster not found — creating {CLUSTER_NAME} ({CLUSTER_SIZE})...")

    compute = AmlCompute(
        name=CLUSTER_NAME,
        size=CLUSTER_SIZE,
        min_instances=0,
        max_instances=4,
        idle_time_before_scale_down=120,
    )
    with_retry(
        lambda: client.compute.begin_create_or_update(compute).result(),
        label="Create compute"
    )
    log(f"Compute created: {CLUSTER_NAME}")
    return CLUSTER_NAME


# ── Environment ───────────────────────────────────────────────────────
def ensure_environment(client):
    log_section("Environment")

    env_yml = PROJECT_ROOT / "environment.yml"
    if not env_yml.exists():
        log(f"X environment.yml not found at: {env_yml}")
        raise FileNotFoundError(f"environment.yml not found: {env_yml}")
    log(f"environment.yml found: {env_yml}")

    env = Environment(
        name=ENV_NAME,
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
        conda_file=str(env_yml),
    )
    registered = with_retry(
        lambda: client.environments.create_or_update(env),
        label="Create environment"
    )
    log(f"Environment ready: {registered.name}:{registered.version}")
    return f"{registered.name}:{registered.version}"


# ── Pipeline ──────────────────────────────────────────────────────────
def build_pipeline(data_path, compute, env):
    log_section("Building Pipeline")

    preprocess = command(
        name="preprocess",
        display_name="Step 1 - Preprocess",
        code="./src",
        command="python preprocess.py --input ${{inputs.raw_data}} --output ${{outputs.processed}}",
        inputs={"raw_data": Input(type=AssetTypes.URI_FILE)},
        outputs={"processed": Output(type=AssetTypes.URI_FOLDER, mode="rw_mount")},
        environment=env,
        compute=compute,
    )

    train = command(
        name="train",
        display_name="Step 2 - Train",
        code="./src",
        command="python train.py --data ${{inputs.processed_data}} --output ${{outputs.model}}",
        inputs={"processed_data": Input(type=AssetTypes.URI_FOLDER)},
        outputs={"model": Output(type=AssetTypes.URI_FOLDER, mode="rw_mount")},
        environment=env,
        compute=compute,
    )

    @dsl.pipeline(name="edt-training-pipeline", description="EDT model training pipeline")
    def pipeline(raw_data):
        prep_step = preprocess(raw_data=raw_data)
        train_step = train(processed_data=prep_step.outputs.processed)
        return {"model": train_step.outputs.model}

    log("Pipeline definition built successfully")
    return pipeline(raw_data=Input(type=AssetTypes.URI_FILE, path=data_path))


# ── Model registration ────────────────────────────────────────────────
def register_model_after_job(client, job_name):
    log_section("Model Registration")

    model = Model(
        name=MODEL_NAME,
        path=f"azureml://jobs/{job_name}/outputs/model",
        type=AssetTypes.CUSTOM_MODEL,
        description=f"EDT model from job {job_name}",
        tags={"market": "japan", "framework": "scikit-learn"},
    )
    registered = with_retry(
        lambda: client.models.create_or_update(model),
        label="Register model"
    )
    log(f"Model registered: {registered.name}:{registered.version}")


# ── Args ──────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Submit EDT training pipeline to Azure ML")
    parser.add_argument("--experiment", default=EXPERIMENT, help="Experiment name")
    parser.add_argument("--no_wait", action="store_true", help="Submit and exit without waiting")
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────
def main():
    log_section("EDT Training Pipeline Submission")
    args = parse_args()
    client = get_ml_client()
    data_path = ensure_data_asset(client)
    compute = ensure_compute(client)
    env = ensure_environment(client)
    pipeline_job = build_pipeline(data_path, compute, env)

    log_section("Submitting Pipeline")
    job = with_retry(
        lambda: client.jobs.create_or_update(pipeline_job, experiment_name=args.experiment),
        label="Submit pipeline"
    )
    log(f"Job submitted : {job.name}")
    log(f"Studio URL    : {job.studio_url}")

    if args.no_wait:
        log("--no_wait set — exiting. Job is running in Azure ML.")
        return

    log("Streaming logs (Ctrl+C to detach)...")
    try:
        client.jobs.stream(job.name)
    except KeyboardInterrupt:
        log("Detached from log stream. Job is still running in Azure ML.")
        return

    final = client.jobs.get(job.name)
    log_section("Pipeline Result")
    log(f"Status: {final.status}")

    if final.status == "Completed":
        register_model_after_job(client, job.name)
        log("Pipeline completed successfully.")
    else:
        log(f"Pipeline failed with status: {final.status}")
        log(f"Check full logs at: {final.studio_url}")
        sys.exit(1)


if __name__ == "__main__":
    main()