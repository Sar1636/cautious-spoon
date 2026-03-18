import os
import argparse
from azure.identity import AzureCliCredential, WorkloadIdentityCredential, ClientSecretCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes


# ── Credential ────────────────────────────────────────────────────────
def get_credential():
    """
    GitHub Actions : WorkloadIdentityCredential (federated OIDC, no secret)
    Local dev      : AzureCliCredential with explicit path search,
                     falls back to ClientSecretCredential if env vars are set
    """
    client_id = os.environ.get("AZURE_CLIENT_ID")
    tenant_id = os.environ.get("AZURE_TENANT_ID")
    token_file = os.environ.get("AZURE_FEDERATED_TOKEN_FILE")

    # GitHub Actions — federated OIDC
    if client_id and tenant_id and token_file:
        print("Auth: WorkloadIdentityCredential (federated OIDC)")
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
            print(f"Auth: AzureCliCredential (path: {path or 'default'})")
            return cred
        except Exception:
            continue

    # Local fallback — ClientSecretCredential from env vars
    client_secret = os.environ.get("AZURE_CLIENT_SECRET")
    if client_id and tenant_id and client_secret:
        print("Auth: ClientSecretCredential (from env vars)")
        return ClientSecretCredential(
            tenant_id=tenant_id,
            client_id=client_id,
            client_secret=client_secret,
        )

    raise EnvironmentError(
        "No valid credential found. Options:\n"
        "  1. Install Azure CLI and run 'az login'\n"
        "  2. Set AZURE_CLIENT_ID, AZURE_TENANT_ID, AZURE_CLIENT_SECRET env vars\n"
        "  Download CLI: https://aka.ms/installazurecliwindows"
    )


# ── Arguments ─────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        default="outputs/edt_model.pkl",
        help="Path to the trained model file or folder"
    )
    parser.add_argument(
        "--model_name",
        default="edt-model-japan",
        help="Name to register the model under in Azure ML"
    )
    parser.add_argument(
        "--description",
        default="EDT model for Japan pizza delivery",
        help="Description for the registered model"
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Model version to register (default: auto-increment)"
    )
    return parser.parse_args()


# ── Azure ML Client ───────────────────────────────────────────────────
def get_ml_client():
    subscription_id = os.environ.get("AZURE_SUBSCRIPTION_ID")
    resource_group = os.environ.get("AZURE_RESOURCE_GROUP")
    workspace_name = os.environ.get("AZURE_ML_WORKSPACE")

    if not subscription_id or not resource_group or not workspace_name:
        raise ValueError(
            "AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, and AZURE_ML_WORKSPACE must be set"
        )

    client = MLClient(
        credential=get_credential(),
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name,
    )

    print(f"Connected to workspace: {client.workspace_name}")
    return client


# ── Auto-increment version ────────────────────────────────────────────
def get_next_version(client, model_name):
    """Find the latest registered version and return the next one."""
    try:
        versions = list(client.models.list(name=model_name))
        if not versions:
            return "1"
        latest = max(int(v.version) for v in versions)
        next_version = str(latest + 1)
        print(f"Latest version: {latest} → registering as version {next_version}")
        return next_version
    except Exception:
        print("Could not determine latest version — defaulting to version 1")
        return "1"


# ── Main ──────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    client = get_ml_client()

    version = args.version if args.version else get_next_version(client, args.model_name)

    print(f"Registering model from: {args.model_path} as version {version}")
    model = Model(
        path=args.model_path,
        name=args.model_name,
        version=version,
        description=args.description,
        type=AssetTypes.CUSTOM_MODEL,
        tags={
            "market": "japan",
            "task": "regression",
            "framework": "scikit-learn"
        },
    )

    registered = client.models.create_or_update(model)
    print("Model registered successfully!")
    print(f"   Name    : {registered.name}")
    print(f"   Version : {registered.version}")
    print(f"   ID      : {registered.id}")


if __name__ == "__main__":
    main()
