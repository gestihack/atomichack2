import os
from pymilvus import MilvusClient, model
client = MilvusClient(os.environ["MILVUS_URL"])

index_params = MilvusClient.prepare_index_params()

# 4.2. Add an index on the vector field.
index_params.add_index(
    field_name="vector",
    metric_type="COSINE",
    index_type="IVF_FLAT",
    index_name="vector_index",
    params={ "nlist": 128 }
)

client.create_index(
    collection_name="demo_collection",
    index_params=index_params
)
client.load_collection("demo_collection")

client.create_index(
    collection_name="guides_collection",
    index_params=index_params
)
client.load_collection("guides_collection")