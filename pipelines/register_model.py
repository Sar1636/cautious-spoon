# """
# register_model.py
# -----------------
# Registers the trained model in the Azure ML Model Registry.
# All credentials are read from environment variables - never hardcoded.
# """

# import os
# import argparse
# import json
# from azure.identity import ClientSecretCredential
# from azure.ai.ml import MLClient
# from azure.ai.ml.entities import Model
# from azure.ai.ml.constants import AssetTypes


# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model_path",    default="outputs/edt_model.pkl")
#     parser.add_argument("--model_name",    default="edt-model-japan")
#     parser.add_argument("--model_version", default="1")
#     parser.add_argument("--description",   default="EDT model for Japan pizza delivery")
#     return parser.parse_args()


# def get_ml_client():
#     # ✅ Read from environment variables — never hardcode
#     tenant_id       = os.environ["AZURE_TENANT_ID"]
#     client_id       = os.environ["AZURE_CLIENT_ID"]
#     client_secret   = os.environ["AZURE_CLIENT_SECRET"]
#     subscription_id = os.environ["AZURE_SUBSCRIPTION_ID"]
#     resource_group  = os.environ.get("AZURE_RESOURCE_GROUP", "rg-edt-japan")
#     workspace_name  = os.environ.get("AZURE_ML_WORKSPACE",   "edt-ml-workspace")

#     print("=" * 50)
#     print(f"TENANT_ID       = {tenant_id}")
#     print(f"SUBSCRIPTION_ID = {subscription_id}")
#     print(f"CLIENT_ID       = {client_id}")
#     print(f"RESOURCE_GROUP  = {resource_group}")
#     print(f"WORKSPACE_NAME  = {workspace_name}")
#     print("=" * 50)

#     credential = ClientSecretCredential(
#         tenant_id     = tenant_id,
#         client_id     = client_id,
#         client_secret = client_secret,
#     )

#     client = MLClient(
#         credential        = credential,
#         subscription_id   = subscription_id,
#         resource_group_name = resource_group,
#         workspace_name    = workspace_name,
#     )

#     print(f"\nConnected: workspace={client.workspace_name}")
#     return client


# def main():
#     args = parse_args()

#     print("\nAuthenticating...")
#     ml_client = get_ml_client()

#     # Verify connection
#     print("\nVerifying workspace connection...")
#     ws = ml_client.workspaces.get(ml_client.workspace_name)
#     print(f"Connected to workspace : {ws.name}")
#     print(f"Resource group         : {ws.resource_group}")
#     print(f"Location               : {ws.location}")

#     # Register model
#     print(f"\nRegistering model from: {args.model_path}")
#     model = Model(
#         path        = args.model_path,
#         name        = args.model_name,
#         description = args.description,
#         type        = AssetTypes.CUSTOM_MODEL,
#         tags        = {
#             "market":    "japan",
#             "task":      "regression",
#             "framework": "scikit-learn",
#         },
#     )

#     registered = ml_client.models.create_or_update(model)
#     print(f"\n✅ Model registered successfully!")
#     print(f"   Name    : {registered.name}")
#     print(f"   Version : {registered.version}")
#     print(f"   ID      : {registered.id}")


# if __name__ == "__main__":
#     main()

import os
import argparse
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes

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
    resource_group  = os.environ.get("AZURE_RESOURCE_GROUP")
    workspace_name  = os.environ.get("AZURE_ML_WORKSPACE")

    if not subscription_id or not resource_group or not workspace_name:
        raise ValueError(
            "AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, and AZURE_ML_WORKSPACE must be set"
        )

    credential = DefaultAzureCredential()

    client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name,
    )

    print(f"Connected to workspace: {client.workspace_name}")
    return client

# ── Main ─────────────────────────────────────────────────────────────
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
    print(f"✅ Model registered successfully!")
    print(f"   Name    : {registered.name}")
    print(f"   Version : {registered.version}")
    print(f"   ID      : {registered.id}")

if __name__ == "__main__":
    main()