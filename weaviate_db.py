import weaviate
import weaviate.classes as wvc


LOCAL_PORT = 8081


def create_from_chunks_and_embeddings(
    chunks: list[str], embeddings: list[list[float]], collection_name: str = "CodeChat"
) -> None:
    client = weaviate.connect_to_local(port=LOCAL_PORT)

    try:
        # Start from scratch each time for this POC
        if client.collections.get(collection_name):
            client.collections.delete(collection_name)

        collection = client.collections.create(
            collection_name,
            vectorizer_config=wvc.config.Configure.Vectorizer.none(),
            vector_index_config=wvc.config.Configure.VectorIndex.hnsw(
                distance_metric=wvc.config.VectorDistances.DOT  # select prefered distance metric
            ),
        )

        objs = []
        for chunk, embedding in zip(chunks, embeddings):
            objs.append(
                wvc.data.DataObject(
                    properties={
                        "text": chunk,
                    },
                    vector=embedding,
                )
            )

        collection = client.collections.get(collection_name)
        collection.data.insert_many(objs)  # This uses batching under the hood

    except Exception as e:
        print(e)

    finally:
        client.close()


def query_weaviate_db(
    vector: list[float], collection_name: str = "CodeChat", k: int = 3
) -> list[str]:
    client = weaviate.connect_to_local(port=LOCAL_PORT)

    try:
        collection = client.collections.get(collection_name)

        response = collection.query.near_vector(
            near_vector=vector,
            limit=k,
            return_metadata=wvc.query.MetadataQuery(certainty=True),
        )

        return [r.properties["text"] for r in response.objects]

    except Exception as e:
        print(e)

    finally:
        client.close()
