"""
submit_pipeline.py
------------------
Submits the EDT training pipeline to Azure ML.

Authentication:
  - GitHub Actions : WorkloadIdentityCredential (federated OIDC — no client secret)
  - Local dev      : AzureCliCredential (uses az login, bypasses env var credentials)
"""

import os
import sys
import time
import argparse
from pathlib import Path

from azure.identity import AzureCliCredential, WorkloadIdentityCredential, ClientSecretCredential
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
    GitHub Actions : WorkloadIdentityCredential (federated OIDC, no secret)
    Local dev      : AzureCliCredential with common Windows path search,
                     falls back to ClientSecretCredential if env vars are set
    """
    client_id = os.environ.get("AZURE_CLIENT_ID")
    tenant_id = os.environ.get("AZURE_TENANT_ID")
    token_file = os.environ.get("AZURE_FEDERATED_TOKEN_FILE")

    # GitHub Actions — federated OIDC
    if client_id and tenant_id and token_file:
        log("Auth: WorkloadIdentityCredential (federated OIDC)")
        return WorkloadIdentityCredential(
            client_id=client_id,
            tenant_id=tenant_id,
        )

    # Local — try AzureCliCredential with common Windows install paths
    cli_paths = [
        None,
        r"C:\Program Files (x86)\Microsoft SDKs\Azure\CLI2\wbin",
        r"C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin",
        os.path.expanduser(r"~\AppData\Local\Programs\Azure CLI\wbin"),
    ]
    for path in cli_paths:
        try:
            if path and path not in os.environ.get("PATH", ""):
                os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")
            cred = AzureCliCredential()
            cred.get_token("https://management.azure.com/.default")
            log(f"Auth: AzureCliCredential (path: {path or 'default'})")
            return cred
        except Exception:
            continue

    # Local fallback — ClientSecretCredential (secret value, not secret ID)
    client_secret = os.environ.get("AZURE_CLIENT_SECRET")
    if client_id and tenant_id and client_secret:
        log("Auth: ClientSecretCredential (from env vars)")
        return ClientSecretCredential(
            tenant_id=tenant_id,
            client_id=client_id,
            client_secret=client_secret,
        )

    raise EnvironmentError(
        "No valid credential found. Options:\n"
        "  1. Install Azure CLI and run 'az login'\n"
        "  2. Set AZURE_CLIENT_ID, AZURE_TENANT_ID, AZURE_CLIENT_SECRET env vars\n"
        "     (AZURE_CLIENT_SECRET must be the secret VALUE, not the secret ID)\n"
        "  Download CLI: https://aka.ms/installazurecliwindows"
    )


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


