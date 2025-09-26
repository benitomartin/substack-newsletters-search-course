import asyncio
import gc
import hashlib
import time
import uuid
from collections.abc import AsyncGenerator
from datetime import datetime
from functools import partial

import numpy as np
import requests
from fastembed import SparseTextEmbedding, TextEmbedding
from huggingface_hub import InferenceClient
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import (
    Snowball,
    SnowballLanguage,
    SnowballParams,
    TextIndexParams,
    TextIndexType,
    TokenizerType,
)
from qdrant_client.models import Batch, Distance, SparseVector, models
from sqlalchemy.orm import Session

from src.config import settings
from src.models.sql_models import SubstackArticle
from src.models.vectorstore_models import ArticleChunkPayload
from src.utils.logger_util import log_batch_status, setup_logging
from src.utils.text_splitter import TextSplitter


class AsyncQdrantVectorStore:
    """Manages asynchronous interactions with Qdrant vector store for article ingestion.

    Initializes Qdrant client, embedding models, and configurations for dense and sparse
    vector storage. Handles collection creation, indexing, and ingestion from SQL.

    Attributes:
        client (AsyncQdrantClient): Qdrant client for vector store operations.
        collection_name (str): Name of the Qdrant collection.
        dense_model (TextEmbedding): Model for dense vector embeddings.
        sparse_model (SparseTextEmbedding): Model for sparse vector embeddings.
        splitter (TextSplitter): Utility for splitting article content into chunks.
        logger: Logger instance for tracking operations and errors.

    """

    def __init__(self, cache_dir: str | None = None):
        """Initialize AsyncQdrantVectorStore with Qdrant client and embedding models."""
        vector_db = settings.qdrant

        # -------------------------------
        # Models & configs
        # -------------------------------
        self.dense_model = TextEmbedding(
            model_name=vector_db.dense_model_name,
            cache_dir=cache_dir,  # Only uses cache_dir if provided
        )
        self.sparse_model = SparseTextEmbedding(
            model_name=vector_db.sparse_model_name,
            cache_dir=cache_dir,  # Only uses cache_dir if provided
        )
        self.embedding_size = vector_db.vector_dim
        self.sparse_batch_size = vector_db.sparse_batch_size
        self.article_batch_size = vector_db.article_batch_size
        self.embed_batch_size = vector_db.embed_batch_size
        self.upsert_batch_size = vector_db.upsert_batch_size
        self.max_concurrent = vector_db.max_concurrent

        # -------------------------------
        # Qdrant client & collection
        # -------------------------------
        self.client = AsyncQdrantClient(url=vector_db.url, api_key=vector_db.api_key)
        self.collection_name = vector_db.collection_name
        self.splitter = TextSplitter()
        self.sparse_vectors_config = {
            "Sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)
        }
        self.quantization_config = models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8,
                quantile=0.99,
                always_ram=False,
            )
        )

        # -------------------------------
        # Logging
        # -------------------------------
        self.logger = setup_logging()
        self.log_batch_status = partial(log_batch_status, self.logger)

        # -------------------------------
        # Jina settings (optional)
        # -------------------------------
        self.jina_settings = settings.jina
        self.use_jina = False  # Set to True to enable Jina integration

        # -------------------------------
        # Hugging Face settings (optional)
        # -------------------------------
        self.hugging_face_settings = settings.hugging_face

        self.hf_client = InferenceClient(
            provider="auto",
            api_key=self.hugging_face_settings.api_key,
        )
        self.hf_model = self.hugging_face_settings.model
        self.use_hf = False  # Set to True to enable HF integration

    # -----------------------
    # Collection management
    # -----------------------

    async def create_collection(self) -> None:
        """Create Qdrant collection if it does not exist.

        Checks for existing collection and creates a new one with dense and sparse vector
        configurations if needed. Logs errors and skips if collection exists.

        Returns:
            None

        Raises:
            RuntimeError: If collection creation fails.
            Exception: For unexpected errors.

        """
        try:
            exists = await self.client.get_collection(collection_name=self.collection_name)
            if exists:
                self.logger.info(
                    f"Collection '{self.collection_name}' already exists. Skipping creation."
                )
                return
        except UnexpectedResponse as e:
            if e.status_code == 404:
                self.logger.info(
                    f"Collection '{self.collection_name}' does not exist. Will create it."
                )
            else:
                self.logger.error(f"Unexpected Qdrant error: {e}")
                raise RuntimeError("Failed to check collection existence") from e

        try:
            self.logger.info(f"Creating Qdrant collection: {self.collection_name}")
            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "Dense": models.VectorParams(size=self.embedding_size, distance=Distance.COSINE)
                },
                sparse_vectors_config=self.sparse_vectors_config,
                quantization_config=self.quantization_config,
                hnsw_config=models.HnswConfigDiff(m=0),
                optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0),
            )
            self.logger.info(f"Collection '{self.collection_name}' created successfully.")
        except Exception as e:
            self.logger.error(f"Failed to create collection '{self.collection_name}': {e}")
            raise RuntimeError("Error creating Qdrant collection") from e

    async def delete_collection(self) -> None:
        """Delete Qdrant collection after user confirmation.

        Prompts user to confirm deletion to prevent accidental data loss. Logs errors and
        skips if canceled.

        Returns:
            None

        Raises:
            RuntimeError: If collection deletion fails.
            Exception: For unexpected errors.

        """
        confirm = input(
            f"Are you sure you want to DELETE the Qdrant collection "
            f"'{self.collection_name}'? Type 'YES' to confirm: "
        )
        if confirm != "YES":
            self.logger.info(f"Deletion of collection '{self.collection_name}' canceled by user.")
            return

        try:
            self.logger.info(f"Deleting Qdrant collection: {self.collection_name}")
            await self.client.delete_collection(collection_name=self.collection_name)
            self.logger.info(f"Qdrant collection '{self.collection_name}' deleted.")
        except Exception as e:
            self.logger.error(f"Failed to delete collection '{self.collection_name}': {e}")
            raise RuntimeError("Error deleting Qdrant collection") from e

    # -------------------------------
    # Update collection to enable HNSW
    # -------------------------------
    async def enable_hnsw(self, m: int = 16, indexing_threshold: int = 20000) -> None:
        """Enable HNSW indexing for the Qdrant collection.

        Updates collection to enable HNSW graph with specified parameters.

        Args:
            m (int, optional): HNSW graph connectivity parameter. Defaults to 16.
            indexing_threshold (int, optional): Threshold for indexing. Defaults to 20000.

        Returns:
            None

        Raises:
            RuntimeError: If HNSW update fails.
            Exception: For unexpected errors.

        """
        try:
            self.logger.info(
                f"Enabling HNSW for collection '{self.collection_name}' "
                f"with m={m} and indexing_threshold={indexing_threshold}"
            )
            await self.client.update_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "Dense": models.VectorParamsDiff(hnsw_config=models.HnswConfigDiff(m=m))
                },
                hnsw_config=models.HnswConfigDiff(m=m),
                optimizers_config=models.OptimizersConfigDiff(
                    indexing_threshold=indexing_threshold
                ),
            )
            self.logger.info(f"HNSW enabled for collection '{self.collection_name}'")
        except Exception as e:
            self.logger.error(f"Failed to enable HNSW for collection '{self.collection_name}': {e}")
            raise RuntimeError("Error enabling HNSW indexing") from e

    # -----------------------
    # Indexes
    # -----------------------
    async def create_feed_author_index(self) -> None:
        """Create keyword index for feed_author field.

        Returns:
            None

        Raises:
            RuntimeError: If index creation fails.
            Exception: For unexpected errors.

        """
        try:
            self.logger.info(f"Creating feed_author index for '{self.collection_name}'")
            await self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="feed_author",
                field_schema=models.KeywordIndexParams(type=models.KeywordIndexType.KEYWORD),
            )
            self.logger.info(f"feed_author index created for '{self.collection_name}'")
        except Exception as e:
            self.logger.error(f"Failed to create feed_author index: {e}")
            raise RuntimeError("Error creating feed_author index") from e

    async def create_article_authors_index(self) -> None:
        """Create keyword index for article_authors field.

        Returns:
            None

        Raises:
            RuntimeError: If index creation fails.
            Exception: For unexpected errors.

        """
        try:
            self.logger.info(f"Creating article_authors index for '{self.collection_name}'")
            await self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="article_authors",
                field_schema=models.KeywordIndexParams(type=models.KeywordIndexType.KEYWORD),
            )
            self.logger.info(f"article_authors index created for '{self.collection_name}'")
        except Exception as e:
            self.logger.error(f"Failed to create article_authors index: {e}")
            raise RuntimeError("Error creating article_authors index") from e

    async def create_article_feed_name_index(self) -> None:
        """Create keyword index for feed_name field.

        Returns:
            None

        Raises:
            RuntimeError: If index creation fails.
            Exception: For unexpected errors.

        """
        try:
            self.logger.info(f"Creating feed_name index for '{self.collection_name}'")
            await self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="feed_name",
                field_schema=models.KeywordIndexParams(type=models.KeywordIndexType.KEYWORD),
            )
            self.logger.info(f"feed_name index created for '{self.collection_name}'")
        except Exception as e:
            self.logger.error(f"Failed to create feed_name index: {e}")
            raise RuntimeError("Error creating feed_name index") from e

    async def create_title_index(self) -> None:
        """Create text index for title field with Snowball stemmer.

        Returns:
            None

        Raises:
            RuntimeError: If index creation fails.
            Exception: For unexpected errors.

        """
        try:
            self.logger.info(f"Creating title index for '{self.collection_name}'")
            await self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="title",
                field_schema=TextIndexParams(
                    type=TextIndexType.TEXT,
                    tokenizer=TokenizerType.WORD,
                    stopwords=models.Language.ENGLISH,
                    lowercase=True,
                    phrase_matching=False,
                    stemmer=SnowballParams(
                        type=Snowball.SNOWBALL, language=SnowballLanguage.ENGLISH
                    ),
                ),
            )
            self.logger.info(f"title index created for '{self.collection_name}'")
        except Exception as e:
            self.logger.error(f"Failed to create title index: {e}")
            raise RuntimeError("Error creating title index") from e

    # -----------------------
    # Embeddings
    # -----------------------

    def jina_dense_vectors(self, texts: list[str]) -> list[list[float]]:
        """Generate dense vectors using Jina API.

        Args:
            texts (list[str]): List of text strings to embed.

        Returns:
            list[list[float]]: List of dense vector embeddings.

        Raises:
            requests.RequestException: If the Jina API request fails.

        """
        try:
            url = getattr(self, "jina_url", f"{self.jina_settings.url}")
            headers = getattr(
                self,
                "jina_headers",
                {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.jina_settings.api_key}",
                },
            )
            data = {
                "model": f"{self.jina_settings.model}",
                "task": "retrieval.passage",
                "dimensions": self.embedding_size,
                "input": texts,
            }
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            return [item["embedding"] for item in response.json().get("data", [])]
        except requests.RequestException as e:
            self.logger.error(f"Jina API request failed: {e}")
            raise

    def hf_dense_vectors(self, texts: list[str]) -> list[list[float]]:
        """Generate dense vectors using Hugging Face Inference API.

        Args:
            texts (list[str]): List of text strings to embed.

        Returns:
            list[list[float]]: List of dense vector embeddings.

        Raises:
            Exception: If Hugging Face inference fails.

        """
        try:
            vectors = []
            for text in texts:
                arr = self.hf_client.feature_extraction(text, model=self.hf_model)
                vectors.append(arr.tolist() if isinstance(arr, np.ndarray) else arr)
            return vectors
        except Exception as e:
            self.logger.error(f"Hugging Face inference failed: {e}")
            raise

    def dense_vectors(self, texts: list[str]) -> list[list[float]]:
        """Generate dense vectors using configured model (Jina, Hugging Face, or local).

        Args:
            texts (list[str]): List of text strings to embed.

        Returns:
            list[list[float]]: List of dense vector embeddings.

        Raises:
            Exception: If embedding generation fails.

        """
        try:
            if self.use_jina:
                return self.jina_dense_vectors(texts)
            elif self.use_hf:
                return self.hf_dense_vectors(texts)
            return [vec.tolist() for vec in self.dense_model.embed(texts)]
        except Exception as e:
            self.logger.error(f"Failed to generate dense vectors: {e}")
            raise

    def sparse_vectors(self, texts: list[str]) -> list[SparseVector]:
        """Generate sparse vectors using sparse embedding model.

        Args:
            texts (list[str]): List of text strings to embed.

        Returns:
            list[SparseVector]: List of sparse vector embeddings.

        Raises:
            Exception: If sparse embedding generation fails.

        """
        try:
            return [
                SparseVector(indices=se.indices.tolist(), values=se.values.tolist())
                for se in self.sparse_model.embed(texts, batch_size=self.sparse_batch_size)
            ]
        except Exception as e:
            self.logger.error(f"Failed to generate sparse vectors: {e}")
            raise

    # -----------------------
    # Embedding helpers (memory-efficient)
    # -----------------------
    # async def embed_batch_async(
    #     self, texts: list[str]
    # ) -> tuple[list[list[float]], list[SparseVector]]:
    #     """Generate dense and sparse embeddings concurrently for a batch of texts.

    #     Args:
    #         texts (list[str]): List of text strings to embed.

    #     Returns:
    #         tuple[list[list[float]], list[SparseVector]]: Dense and sparse embeddings.

    #     Raises:
    #         RuntimeError: If embedding generation fails.
    #     """
    #     try:
    #         # Run embeddings concurrently in threads
    #         dense_task = asyncio.to_thread(self.dense_model.embed, texts)
    #         sparse_task = asyncio.to_thread(
    #             self.sparse_model.embed, texts, batch_size=self.sparse_batch_size
    #         )
    #         dense_result, sparse_result = await asyncio.gather(dense_task, sparse_task)

    #         # Convert to upsert-friendly format
    #         dense_vecs = [vec.tolist() for vec in dense_result]
    #         sparse_vecs = [SparseVector(indices=se.indices.tolist(),
    #                                      values=se.values.tolist()) for se in sparse_result]

    #         # Free memory
    #         del dense_result, sparse_result
    #         return dense_vecs, sparse_vecs
    #     except Exception as e:
    #         self.logger.error(f"Failed to generate embeddings: {e}")
    #         raise RuntimeError("Error generating batch embeddings") from e

    async def embed_batch_async(
        self, texts: list[str]
    ) -> tuple[list[list[float]], list[SparseVector]]:
        """Generate dense and sparse embeddings concurrently for a batch of texts.

        Args:
            texts (list[str]): List of text strings to embed.

        Returns:
            tuple[list[list[float]], list[SparseVector]]: Dense and sparse embeddings.

        Raises:
            RuntimeError: If embedding generation fails.

        """
        try:
            # Run embeddings concurrently in threads
            dense_task = asyncio.to_thread(self.dense_vectors, texts)  # use dense_vectors() now
            sparse_task = asyncio.to_thread(
                self.sparse_model.embed, texts, batch_size=self.sparse_batch_size
            )
            dense_result, sparse_result = await asyncio.gather(dense_task, sparse_task)

            # Convert to upsert-friendly format
            dense_vecs = [
                vec.tolist() if isinstance(vec, np.ndarray) else vec for vec in dense_result
            ]
            sparse_vecs = [
                SparseVector(indices=se.indices.tolist(), values=se.values.tolist())
                for se in sparse_result
            ]

            # Free memory
            del dense_result, sparse_result
            return dense_vecs, sparse_vecs
        except Exception as e:
            self.logger.error(f"Failed to generate embeddings: {e}")
            raise RuntimeError("Error generating batch embeddings") from e

    async def _article_batch_generator(
        self, session: Session, from_date: datetime | None = None
    ) -> AsyncGenerator[list[SubstackArticle], None]:
        """Yield batches of articles from SQL database.

        Args:
            session (Session): SQLAlchemy session for querying articles.
            from_date (datetime, optional): Filter articles from this date.

        Yields:
            list[SubstackArticle]: Batch of articles.

        Raises:
            Exception: If database query fails.

        """
        # Query is synchronous. For 5 articles ok
        # But concurrent requests may be needed for larger batches (e.g. 100+ articles).
        # In this case change to async the init_session.py
        try:
            offset = 0
            while True:
                query = session.query(SubstackArticle).order_by(SubstackArticle.published_at)
                if from_date:
                    query = query.filter(SubstackArticle.published_at >= from_date)
                articles = query.offset(offset).limit(self.article_batch_size).all()
                if not articles:
                    break
                yield articles
                offset += self.article_batch_size
        except Exception as e:
            self.logger.error(f"Failed to fetch article batch: {e}")
            raise

    async def ingest_from_sql(self, session: Session, from_date: datetime | None = None):
        """Ingest articles from SQL database into Qdrant vector store.

        Fetches articles in batches, generates embeddings, and upserts them to Qdrant.
        Skips existing articles and logs throughput.

        Args:
            session (Session): SQLAlchemy session for querying articles.
            from_date (datetime, optional): Filter articles from this date.

        Returns:
            None

        Raises:
            RuntimeError: If ingestion or upsert fails.
            Exception: For unexpected errors.

        """
        self.logger.info(
            f"Starting ingestion in Qdrant collection '{self.collection_name}' "
            f"from SQL (batch size: {self.article_batch_size})"
        )
        try:
            # Limit concurrency to avoid ingestion overload into Qdrant
            semaphore = asyncio.Semaphore(max(2, self.max_concurrent))
            total_articles = 0
            total_chunks = 0
            start_time = time.time()  # cumulative start time

            async for articles in self._article_batch_generator(session, from_date=from_date):
                all_chunks, all_ids, all_payloads = [], [], []

                for article in articles:
                    chunks = self.splitter.split_text(article.content)
                    ids = [
                        str(
                            uuid.UUID(
                                hashlib.sha256(f"{article.url}_{chunk}".encode()).hexdigest()[:32]
                            )
                        )
                        for chunk in chunks
                    ]
                    payloads = [
                        ArticleChunkPayload(
                            feed_name=article.feed_name,
                            feed_author=article.feed_author,
                            article_authors=article.article_authors,
                            title=article.title,
                            url=article.url,
                            published_at=str(article.published_at),
                            created_at=str(article.created_at),
                            chunk_index=i,
                            chunk_text=chunk,
                        )
                        for i, chunk in enumerate(chunks)
                    ]

                    # Check existing IDs
                    existing_points = await self.client.retrieve(
                        collection_name=self.collection_name, ids=ids
                    )
                    existing_ids = {p.id for p in existing_points}

                    new_chunks = [
                        c for c, id_ in zip(chunks, ids, strict=False) if id_ not in existing_ids
                    ]
                    new_ids = [id_ for id_ in ids if id_ not in existing_ids]
                    new_payloads = [
                        p for p, id_ in zip(payloads, ids, strict=False) if id_ not in existing_ids
                    ]

                    self.logger.info(
                        f"Article '{article.title}': total chunks = {len(chunks)}, "
                        f"existing chunks = {len(existing_ids)}, new chunks = {len(new_chunks)}"
                    )

                    all_chunks.extend(new_chunks)
                    all_ids.extend(new_ids)
                    all_payloads.extend(new_payloads)
                    total_articles += 1

                # -------------------------------
                # Process all chunks in batches
                # -------------------------------
                for start in range(0, len(all_chunks), self.upsert_batch_size):
                    sub_chunks = all_chunks[start : start + self.upsert_batch_size]
                    sub_ids: list[int | str] = all_ids[start : start + self.upsert_batch_size]  # type: ignore
                    sub_payloads = all_payloads[start : start + self.upsert_batch_size]

                    batch_start_time = time.time()  # start time for this batch
                    dense_vecs, sparse_vecs = await self.embed_batch_async(sub_chunks)

                    async with semaphore:
                        await self.client.upsert(
                            collection_name=self.collection_name,
                            points=Batch(
                                ids=sub_ids,  # type: ignore
                                payloads=[p.dict() for p in sub_payloads],
                                vectors={"Dense": dense_vecs, "Sparse": sparse_vecs},  # type: ignore
                            ),
                        )

                    total_chunks += len(sub_chunks)

                    # -------------------------------
                    # Throughput logging
                    # -------------------------------
                    batch_elapsed = time.time() - batch_start_time
                    batch_speed = len(sub_chunks) / batch_elapsed if batch_elapsed > 0 else 0

                    cumulative_elapsed = time.time() - start_time
                    cumulative_speed = (
                        total_chunks / cumulative_elapsed if cumulative_elapsed > 0 else 0
                    )
                    self.log_batch_status(
                        action="Batch ingested",
                        batch_size=len(sub_chunks),
                        total_articles=total_articles,
                        total_chunks=total_chunks,
                    )

                    self.logger.info(
                        f"Batch ingested: {len(sub_chunks)} chunks | "
                        f"Batch speed: {batch_speed:.2f} chunks/sec | "
                        f"Cumulative speed: {cumulative_speed:.2f} chunks/sec | "
                        f"Total articles: {total_articles}, Total chunks: {total_chunks}"
                    )

                    del dense_vecs, sparse_vecs, sub_chunks, sub_ids, sub_payloads
                    gc.collect()

            # -------------------------------
            # Final cumulative average
            # -------------------------------
            final_elapsed = time.time() - start_time
            final_speed = total_chunks / final_elapsed if final_elapsed > 0 else 0
            self.logger.info(
                f"Ingestion complete: {total_articles} articles, {total_chunks} chunks, "
                f"final average speed = {final_speed:.2f} chunks/sec"
            )
        except Exception as e:
            self.logger.error(f"Failed to ingest articles to Qdrant: {e}")
            raise RuntimeError("Error during SQL to Qdrant ingestion") from e
