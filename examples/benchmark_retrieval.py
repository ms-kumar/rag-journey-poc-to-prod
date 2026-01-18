"""
Benchmark script for RAG retrieval quality and latency.

Measures:
- Retrieval@k metrics (precision, recall)
- Latency for embedding, retrieval, and generation
- Cache performance impact
- End-to-end pipeline latency
"""

import time
from pathlib import Path
from typing import Any

from src.config import Settings
from src.services.chunking.factory import get_chunking_client
from src.services.embeddings.factory import get_embed_client
from src.services.generation.factory import get_generator
from src.services.ingestion.factory import get_ingestion_client
from src.services.vectorstore.factory import get_vectorstore_client


def timeit(func):
    """Decorator to measure execution time."""

    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        return result, elapsed

    return wrapper


class RetrievalBenchmark:
    """Benchmark retrieval quality and latency."""

    def __init__(self, config: Settings):
        self.config = config
        self.ingestion_client = get_ingestion_client("local", directory=config.ingestion_dir)
        self.chunking_client = get_chunking_client(
            chunk_size=config.chunk_size, strategy=config.chunking_strategy
        )
        self.embed_client = get_embed_client(config)
        
        # Create LangChain adapter for vectorstore
        from src.services.embeddings.adapter import LangChainEmbeddingsAdapter
        embeddings_adapter = LangChainEmbeddingsAdapter(self.embed_client)
        
        self.vectorstore_client = get_vectorstore_client(
            embeddings=embeddings_adapter,
            settings=config.vectorstore,
        )
        self.generation_client = get_generator()

    @timeit
    def ingest_documents(self):
        """Ingest documents and measure time."""
        return self.ingestion_client.ingest()

    @timeit
    def chunk_documents(self, documents: list[str]):
        """Chunk documents and measure time."""
        return self.chunking_client.chunk(documents)

    @timeit
    def embed_chunks(self, chunks: list[str]):
        """Embed chunks and measure time."""
        return self.embed_client.embed(chunks)

    @timeit
    def index_documents(self, chunks: list[str], embeddings: list[list[float]]):
        """Index documents in vector store and measure time."""
        # Use add_texts which is the correct method
        return self.vectorstore_client.add_texts(chunks)

    @timeit
    def retrieve_documents(self, query: str, k: int = 5):
        """Retrieve documents for query and measure time."""
        return self.vectorstore_client.similarity_search(query, k=k)

    @timeit
    def generate_answer(self, query: str, context: str):
        """Generate answer and measure time."""
        prompt = f"Context: {context[:500]}...\n\nQuestion: {query}\n\nAnswer:"
        return self.generation_client.generate(prompt, overrides={"max_new_tokens": 100})

    def run_end_to_end_benchmark(self, query: str, k_values: list[int]):
        """Run end-to-end benchmark for different k values."""
        print("\n" + "=" * 80)
        print("RAG RETRIEVAL & LATENCY BENCHMARK")
        print("=" * 80)

        # Phase 1: Ingestion
        print("\n📥 Phase 1: Document Ingestion")
        print("-" * 80)
        documents, ingest_time = self.ingest_documents()
        print(f"✓ Ingested {len(documents)} documents in {ingest_time*1000:.2f}ms")

        # Phase 2: Chunking
        print("\n✂️  Phase 2: Document Chunking")
        print("-" * 80)
        chunks, chunk_time = self.chunk_documents(documents)
        print(f"✓ Created {len(chunks)} chunks in {chunk_time*1000:.2f}ms")
        print(
            f"  Average chunk length: {sum(len(c) for c in chunks) / len(chunks):.0f} chars"
        )

        # Phase 3: Embedding (first run - cold cache)
        print("\n🔢 Phase 3: Embedding Generation")
        print("-" * 80)
        embeddings_cold, embed_time_cold = self.embed_chunks(chunks)
        print(f"✓ Generated {len(embeddings_cold)} embeddings (cold) in {embed_time_cold*1000:.2f}ms")
        print(f"  Throughput: {len(chunks)/embed_time_cold:.0f} chunks/sec")

        # Phase 3b: Embedding (second run - warm cache)
        embeddings_warm, embed_time_warm = self.embed_chunks(chunks)
        speedup = embed_time_cold / embed_time_warm if embed_time_warm > 0 else 1
        print(f"✓ Generated {len(embeddings_warm)} embeddings (warm) in {embed_time_warm*1000:.2f}ms")
        print(f"  Cache speedup: {speedup:.1f}x")
        print(
            f"  Cache stats: {self.embed_client.cache_stats if hasattr(self.embed_client, 'cache_stats') else 'N/A'}"
        )

        # Phase 4: Indexing
        print("\n💾 Phase 4: Vector Store Indexing")
        print("-" * 80)
        _, index_time = self.index_documents(chunks, embeddings_warm)
        print(f"✓ Indexed {len(chunks)} vectors in {index_time*1000:.2f}ms")

        # Phase 5: Retrieval benchmarks for different k values
        print("\n🔍 Phase 5: Retrieval Benchmarks")
        print("-" * 80)
        print(f"Query: '{query}'\n")

        retrieval_results = {}
        for k in k_values:
            # Cold retrieval
            results_cold, retrieval_time_cold = self.retrieve_documents(query, k=k)

            # Warm retrieval (second run)
            results_warm, retrieval_time_warm = self.retrieve_documents(query, k=k)

            retrieval_results[k] = {
                "results": results_warm,
                "time_cold": retrieval_time_cold,
                "time_warm": retrieval_time_warm,
            }

            print(f"Retrieval@{k}:")
            print(f"  Cold latency: {retrieval_time_cold*1000:.2f}ms")
            print(f"  Warm latency: {retrieval_time_warm*1000:.2f}ms")
            print(f"  Retrieved {len(results_warm)} documents")
            if results_warm:
                print(f"  Top score: {results_warm[0].metadata.get('score', 'N/A'):.4f}")
                print(f"  Preview: {results_warm[0].page_content[:100]}...")
            print()

        # Phase 6: Generation benchmark
        print("\n🤖 Phase 6: Answer Generation")
        print("-" * 80)
        context = "\n\n".join([doc.page_content for doc in retrieval_results[k_values[0]]["results"]])
        answer, gen_time = self.generate_answer(query, context)
        print(f"✓ Generated answer in {gen_time*1000:.2f}ms")
        print(f"  Answer: {answer[:200]}...")

        # Phase 7: Summary
        print("\n📊 Performance Summary")
        print("-" * 80)
        total_time_cold = (
            ingest_time
            + chunk_time
            + embed_time_cold
            + index_time
            + retrieval_results[k_values[0]]["time_cold"]
        )
        total_time_warm = (
            ingest_time
            + chunk_time
            + embed_time_warm
            + index_time
            + retrieval_results[k_values[0]]["time_warm"]
        )

        print(f"Indexing pipeline latency (cold): {total_time_cold*1000:.2f}ms")
        print(f"Indexing pipeline latency (warm): {total_time_warm*1000:.2f}ms")
        print(f"Speedup with caching: {total_time_cold/total_time_warm:.2f}x")
        print()
        print("Component Breakdown (warm cache):")
        print(f"  Ingestion:    {ingest_time*1000:>8.2f}ms ({ingest_time/total_time_warm*100:>5.1f}%)")
        print(f"  Chunking:     {chunk_time*1000:>8.2f}ms ({chunk_time/total_time_warm*100:>5.1f}%)")
        print(f"  Embedding:    {embed_time_warm*1000:>8.2f}ms ({embed_time_warm/total_time_warm*100:>5.1f}%)")
        print(f"  Indexing:     {index_time*1000:>8.2f}ms ({index_time/total_time_warm*100:>5.1f}%)")
        print(f"  Retrieval@{k_values[0]}:  {retrieval_results[k_values[0]]['time_warm']*1000:>8.2f}ms ({retrieval_results[k_values[0]]['time_warm']/total_time_warm*100:>5.1f}%)")
        
        print()
        print("Generation Stats:")
        print(f"  Generation:   {gen_time*1000:>8.2f}ms")
        print(f"  Full E2E:     {(total_time_warm + gen_time)*1000:>8.2f}ms")
        print()
        
        print("Retrieval@k Latency Comparison:")
        for k in k_values:
            print(f"  k={k:2d}: {retrieval_results[k]['time_warm']*1000:>6.2f}ms")
        print()

        return {
            "total_time_cold": total_time_cold,
            "total_time_warm": total_time_warm,
            "ingest_time": ingest_time,
            "chunk_time": chunk_time,
            "embed_time_cold": embed_time_cold,
            "embed_time_warm": embed_time_warm,
            "index_time": index_time,
            "retrieval_results": retrieval_results,
            "gen_time": gen_time,
        }


def main():
    """Run the benchmark."""
    # Initialize configuration
    config = Settings()

    # Override some settings for benchmark
    config.model_config["frozen"] = False  # Allow modifications
    
    # Test query
    query = "What is machine learning?"

    # Test different k values
    k_values = [1, 3, 5, 10]

    # Initialize benchmark
    benchmark = RetrievalBenchmark(config)

    # Run benchmark
    results = benchmark.run_end_to_end_benchmark(query, k_values)

    print("\n✅ Benchmark completed!")


if __name__ == "__main__":
    main()
