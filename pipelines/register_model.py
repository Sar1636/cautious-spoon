import os
import argparse
from azure.identity import WorkloadIdentityCredential, DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes


# ── Credential ────────────────────────────────────────────────────────
def get_credential():
    client_id = os.environ.get("AZURE_CLIENT_ID")
    tenant_id = os.environ.get("AZURE_TENANT_ID")
    token_file = os.environ.get("AZURE_FEDERATED_TOKEN_FILE")

    if client_id and tenant_id and token_file:
        print("Auth: WorkloadIdentityCredential (federated OIDC)")
        return WorkloadIdentityCredential(
            client_id=client_id,
            tenant_id=tenant_id,
        )

    print("Auth: DefaultAzureCredential (local fallback)")
    return DefaultAzureCredential()


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


# ── Main ──────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    client = get_ml_client()

    print(f"Registering model from: {args.model_path}")
    model = Model(
        path=args.model_path,
        name=args.model_name,
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