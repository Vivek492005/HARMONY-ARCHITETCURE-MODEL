# HARMONY-Core Ingestion Module

The `ingestion` module processes external knowledge files and indexes them into vector and graph databases for reference.

## Component Overview

### 1. Document Parsing
Includes parsers for multiple text and document formats:
- Plaintext files (`.txt`)
- Markdown documents (`.md`)
- Portable Document Format (`.pdf`)
- Word documents (`.docx`)

### 2. `KnowledgeIngestionPipeline`
An automated flow to parse, embed, and index external assets:
- **Chunking**: Splits parsed document text into semantic overlap chunks.
- **Embedding**: Generates dense representations using the `EmbeddingManager`.
- **Indexing**: Upserts generated vectors into `VectorMemory` stores.
- **Incremental Ingestion**: Keeps track of file hashes in a local registry database to skip files that have already been indexed, preventing duplicate database entries.
