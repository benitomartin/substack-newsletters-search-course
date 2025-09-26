# Changelog

## 1.0.0

Released on September 16, 2024.

### Added

- Initial release of the Substack Articles Search Engine
- Utilizes Prefect for workflow orchestration and scheduling
- Fetches articles from Substack newsletters using RSS and feeds
- Processes and cleans article content and metadata and ingests into a Supabase SQL table
- Ingest articles from the Supabase SQL table into a Qdrant vector store collection
- Implements a backend API using FastAPI to handle search queries and serve results
- Deploys the application on Google Cloud Run for scalability.
- Provides a Gradio-based user interface for searching and displaying articles
- Includes CI/CD pipelines for automated testing and deployment.
