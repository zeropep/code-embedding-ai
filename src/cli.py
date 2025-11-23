#!/usr/bin/env python3
"""
Code Embedding AI Pipeline - Command Line Interface

A CLI tool for processing source code repositories to generate semantic embeddings
for code search and analysis.
"""

import click
import asyncio
import json
import time
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import structlog

from .embeddings.embedding_pipeline import EmbeddingPipeline
from .code_parser.models import ParserConfig
from .security.models import SecurityConfig
from .embeddings.models import EmbeddingConfig
from .database.models import VectorDBConfig
from .database.vector_store import VectorStore
from .updates.update_service import UpdateService
from .updates.models import UpdateConfig, UpdateRequest


# Configure logging for CLI
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
    ],
    wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO level
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


@click.group()
@click.option("--config", default=None, help="Path to configuration file")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.pass_context
def cli(ctx, config, verbose):
    """Code Embedding AI Pipeline - Semantic code analysis and search"""
    ctx.ensure_object(dict)

    if verbose:
        # Enable debug logging
        structlog.configure(
            wrapper_class=structlog.make_filtering_bound_logger(10)  # DEBUG level
        )

    ctx.obj['config_file'] = config
    ctx.obj['verbose'] = verbose

    click.echo("🤖 Code Embedding AI Pipeline CLI")
    if config:
        click.echo(f"📁 Using config: {config}")


@cli.group()
def process():
    """Commands for processing repositories and files"""
    pass


@process.command("repository")
@click.argument("repo_path", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--output-db", default="./embeddings.db", help="Output ChromaDB path")
@click.option("--force", is_flag=True, help="Force reprocessing of all files")
@click.option("--include", multiple=True, help="File patterns to include")
@click.option("--exclude", multiple=True, help="File patterns to exclude")
@click.option("--no-security", is_flag=True, help="Disable security scanning")
@click.option("--batch-size", default=20, help="Batch size for processing")
@click.pass_context
def process_repository(ctx, repo_path, output_db, force, include, exclude, no_security, batch_size):
    """Process an entire repository to generate embeddings"""
    async def _process():
        try:
            click.echo(f"🔄 Processing repository: {repo_path}")
            start_time = time.time()

            # Create configurations
            parser_config = ParserConfig()
            security_config = SecurityConfig(enabled=not no_security)
            embedding_config = EmbeddingConfig(batch_size=batch_size)

            # Initialize pipeline
            pipeline = EmbeddingPipeline(
                parser_config=parser_config,
                security_config=security_config,
                embedding_config=embedding_config
            )

            # Process repository
            result = await pipeline.process_repository(repo_path)

            processing_time = time.time() - start_time

            if result["status"] == "success":
                click.echo("✅ Repository processing completed successfully!")

                # Display summary
                summary = result.get("processing_summary", {})
                click.echo(f"📊 Summary:")
                click.echo(f"  • Files processed: {summary.get('total_files_parsed', 0)}")
                click.echo(f"  • Chunks created: {summary.get('total_chunks_created', 0)}")
                click.echo(f"  • Embeddings generated: {summary.get('chunks_with_embeddings', 0)}")
                click.echo(f"  • Processing time: {processing_time:.2f}s")

                # Display security stats if enabled
                if not no_security and "security_stats" in result:
                    sec_stats = result["security_stats"]["scan_summary"]
                    click.echo(f"🔒 Security:")
                    click.echo(f"  • Secrets found: {sec_stats.get('total_secrets_found', 0)}")
                    click.echo(f"  • Files with secrets: {sec_stats.get('files_with_secrets', 0)}")

            else:
                click.echo(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
                sys.exit(1)

        except Exception as e:
            click.echo(f"❌ Error: {str(e)}")
            logger.error("Repository processing failed", error=str(e))
            sys.exit(1)

    asyncio.run(_process())


@process.command("files")
@click.argument("files", nargs=-1, required=True, type=click.Path(exists=True, file_okay=True))
@click.option("--output-db", default="./embeddings.db", help="Output ChromaDB path")
@click.pass_context
def process_files(ctx, files, output_db):
    """Process specific files"""
    async def _process():
        try:
            click.echo(f"🔄 Processing {len(files)} file(s)")

            pipeline = EmbeddingPipeline()
            result = await pipeline.process_files(list(files))

            if result["status"] == "success":
                click.echo("✅ Files processed successfully!")
                summary = result.get("processing_summary", {})
                click.echo(f"📊 Chunks created: {summary.get('total_chunks_created', 0)}")
                click.echo(f"📊 Embeddings: {summary.get('chunks_with_embeddings', 0)}")
            else:
                click.echo(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
                sys.exit(1)

        except Exception as e:
            click.echo(f"❌ Error: {str(e)}")
            sys.exit(1)

    asyncio.run(_process())


@cli.group()
def search():
    """Commands for searching embeddings"""
    pass


@search.command("semantic")
@click.argument("query")
@click.option("--db-path", default="./embeddings.db", help="ChromaDB path")
@click.option("--limit", default=10, help="Number of results")
@click.option("--min-similarity", default=0.0, help="Minimum similarity score")
@click.option("--show-content", is_flag=True, help="Show code content")
@click.option("--format", "output_format", type=click.Choice(["table", "json", "simple"]), default="table", help="Output format")
def semantic_search(query, db_path, limit, min_similarity, show_content, output_format):
    """Search for semantically similar code"""
    async def _search():
        try:
            click.echo(f"🔍 Searching for: {query}")

            # Initialize vector store
            vector_config = VectorDBConfig(persist_directory=db_path)
            vector_store = VectorStore(vector_config)

            if not vector_store.connect():
                click.echo("❌ Failed to connect to vector database")
                sys.exit(1)

            # For demonstration, we'll show a mock search
            # In reality, you'd need the embedding service to generate query embeddings
            click.echo("⚠️  Note: Full semantic search requires running API server")
            click.echo("💡 Use metadata search for direct database queries")

            # Show some sample results from database
            stats = vector_store.get_statistics()
            click.echo(f"📊 Database contains {stats.total_chunks} chunks from {stats.total_files} files")

        except Exception as e:
            click.echo(f"❌ Search error: {str(e)}")
            sys.exit(1)

    asyncio.run(_search())


@search.command("metadata")
@click.option("--file-path", help="Filter by file path")
@click.option("--function-name", help="Filter by function name")
@click.option("--class-name", help="Filter by class name")
@click.option("--layer-type", help="Filter by layer type (Controller, Service, etc.)")
@click.option("--language", help="Filter by language")
@click.option("--db-path", default="./embeddings.db", help="ChromaDB path")
@click.option("--limit", default=10, help="Number of results")
@click.option("--format", "output_format", type=click.Choice(["table", "json", "simple"]), default="table", help="Output format")
def metadata_search(file_path, function_name, class_name, layer_type, language, db_path, limit, output_format):
    """Search by metadata filters"""
    async def _search():
        try:
            # Build filters
            filters = {}
            if file_path:
                filters["file_path"] = file_path
            if function_name:
                filters["function_name"] = function_name
            if class_name:
                filters["class_name"] = class_name
            if layer_type:
                filters["layer_type"] = layer_type
            if language:
                filters["language"] = language

            if not filters:
                click.echo("❌ At least one filter must be specified")
                sys.exit(1)

            click.echo(f"🔍 Searching with filters: {filters}")

            # Initialize vector store
            vector_config = VectorDBConfig(persist_directory=db_path)
            vector_store = VectorStore(vector_config)

            if not vector_store.connect():
                click.echo("❌ Failed to connect to vector database")
                sys.exit(1)

            # Perform metadata search
            results = vector_store.search_by_metadata(filters, limit=limit)

            click.echo(f"✅ Found {len(results)} results")

            if output_format == "json":
                output = [
                    {
                        "chunk_id": r.chunk_id,
                        "file_path": r.metadata.get("file_path", ""),
                        "function_name": r.metadata.get("function_name"),
                        "class_name": r.metadata.get("class_name"),
                        "layer_type": r.metadata.get("layer_type"),
                        "lines": f"{r.metadata.get('start_line', 0)}-{r.metadata.get('end_line', 0)}"
                    }
                    for r in results
                ]
                click.echo(json.dumps(output, indent=2))

            elif output_format == "table":
                if results:
                    click.echo(f"\n{'File Path':<40} {'Function':<20} {'Class':<15} {'Layer':<10} {'Lines':<8}")
                    click.echo("-" * 95)

                    for result in results:
                        file_path = result.metadata.get("file_path", "")[:38] + ".." if len(result.metadata.get("file_path", "")) > 40 else result.metadata.get("file_path", "")
                        function_name = result.metadata.get("function_name", "")[:18] if result.metadata.get("function_name") else ""
                        class_name = result.metadata.get("class_name", "")[:13] if result.metadata.get("class_name") else ""
                        layer_type = result.metadata.get("layer_type", "")[:8]
                        lines = f"{result.metadata.get('start_line', 0)}-{result.metadata.get('end_line', 0)}"

                        click.echo(f"{file_path:<40} {function_name:<20} {class_name:<15} {layer_type:<10} {lines:<8}")

            else:  # simple format
                for i, result in enumerate(results, 1):
                    click.echo(f"{i}. {result.metadata.get('file_path', '')}:{result.metadata.get('start_line', 0)}")
                    if result.metadata.get("function_name"):
                        click.echo(f"   Function: {result.metadata.get('function_name')}")
                    if result.metadata.get("class_name"):
                        click.echo(f"   Class: {result.metadata.get('class_name')}")

        except Exception as e:
            click.echo(f"❌ Search error: {str(e)}")
            sys.exit(1)

    asyncio.run(_search())


@cli.group()
def update():
    """Commands for incremental updates"""
    pass


@update.command("check")
@click.argument("repo_path", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--state-dir", default="./update_state", help="Update state directory")
def check_updates(repo_path, state_dir):
    """Check for updates without applying them"""
    async def _check():
        try:
            click.echo(f"🔍 Checking for updates in: {repo_path}")

            update_service = UpdateService(repo_path, state_dir)
            await update_service.start()

            try:
                result = await update_service.quick_update()

                click.echo(f"📊 Update check completed:")
                click.echo(f"  • Status: {result.status.value}")
                click.echo(f"  • Changes detected: {result.total_changes}")
                click.echo(f"  • Files processed: {result.files_processed}")

                if result.total_changes > 0:
                    click.echo(f"  • Chunks added: {result.chunks_added}")
                    click.echo(f"  • Chunks updated: {result.chunks_updated}")
                    click.echo(f"  • Chunks deleted: {result.chunks_deleted}")

            finally:
                await update_service.stop()

        except Exception as e:
            click.echo(f"❌ Update check failed: {str(e)}")
            sys.exit(1)

    asyncio.run(_check())


@update.command("apply")
@click.argument("repo_path", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--state-dir", default="./update_state", help="Update state directory")
@click.option("--force", is_flag=True, help="Force full update")
def apply_updates(repo_path, state_dir, force):
    """Apply incremental updates"""
    async def _apply():
        try:
            mode = "full" if force else "incremental"
            click.echo(f"🔄 Applying {mode} update to: {repo_path}")

            update_service = UpdateService(repo_path, state_dir)
            await update_service.start()

            try:
                if force:
                    result = await update_service.force_full_update()
                else:
                    result = await update_service.quick_update()

                if result.status.value == "completed":
                    click.echo("✅ Update completed successfully!")
                    click.echo(f"📊 Summary:")
                    click.echo(f"  • Files processed: {result.files_processed}")
                    click.echo(f"  • Chunks added: {result.chunks_added}")
                    click.echo(f"  • Chunks updated: {result.chunks_updated}")
                    click.echo(f"  • Chunks deleted: {result.chunks_deleted}")
                    click.echo(f"  • Processing time: {result.processing_time:.2f}s")
                else:
                    click.echo(f"❌ Update failed: {result.error_message}")
                    sys.exit(1)

            finally:
                await update_service.stop()

        except Exception as e:
            click.echo(f"❌ Update failed: {str(e)}")
            sys.exit(1)

    asyncio.run(_apply())


@cli.group()
def server():
    """Server management commands"""
    pass


@server.command("start")
@click.option("--host", default="localhost", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development")
def start_server(host, port, reload):
    """Start the REST API server"""
    import uvicorn

    click.echo(f"🚀 Starting Code Embedding AI API server")
    click.echo(f"🌐 Server will be available at: http://{host}:{port}")
    click.echo(f"📖 API documentation: http://{host}:{port}/docs")

    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


@cli.group()
def db():
    """Database management commands"""
    pass


@db.command("stats")
@click.option("--db-path", default="./embeddings.db", help="ChromaDB path")
def show_stats(db_path):
    """Show database statistics"""
    try:
        vector_config = VectorDBConfig(persist_directory=db_path)
        vector_store = VectorStore(vector_config)

        if not vector_store.connect():
            click.echo("❌ Failed to connect to vector database")
            sys.exit(1)

        stats = vector_store.get_statistics()

        click.echo("📊 Database Statistics:")
        click.echo(f"  • Total chunks: {stats.total_chunks:,}")
        click.echo(f"  • Total files: {stats.total_files:,}")
        click.echo(f"  • Collection size: {stats.collection_size_mb:.2f} MB")

        if stats.language_counts:
            click.echo(f"  • Languages:")
            for lang, count in stats.language_counts.items():
                click.echo(f"    - {lang}: {count:,}")

        if stats.layer_counts:
            click.echo(f"  • Layers:")
            for layer, count in stats.layer_counts.items():
                click.echo(f"    - {layer}: {count:,}")

    except Exception as e:
        click.echo(f"❌ Database error: {str(e)}")
        sys.exit(1)


@db.command("reset")
@click.option("--db-path", default="./embeddings.db", help="ChromaDB path")
@click.confirmation_option(prompt="⚠️  This will delete all data. Continue?")
def reset_database(db_path):
    """Reset the vector database (delete all data)"""
    try:
        vector_config = VectorDBConfig(persist_directory=db_path)
        vector_store = VectorStore(vector_config)

        if not vector_store.connect():
            click.echo("❌ Failed to connect to vector database")
            sys.exit(1)

        success = vector_store.reset_database()

        if success:
            click.echo("✅ Database reset successfully")
        else:
            click.echo("❌ Database reset failed")
            sys.exit(1)

    except Exception as e:
        click.echo(f"❌ Database error: {str(e)}")
        sys.exit(1)


@cli.command("config")
@click.option("--show", is_flag=True, help="Show current configuration")
def config_command(show):
    """Configuration management"""
    if show:
        click.echo("📋 Current Configuration:")
        click.echo("  Parser:")
        click.echo("    • Min tokens: 50")
        click.echo("    • Max tokens: 500")
        click.echo("    • Supported extensions: .java, .kt, .html, .xml, .yml, .yaml, .properties")
        click.echo("  Security:")
        click.echo("    • Secret scanning: enabled")
        click.echo("    • Sensitivity threshold: 0.7")
        click.echo("  Embedding:")
        click.echo("    • Model: jina-embeddings-v2-base-code")
        click.echo("    • Batch size: 20")
        click.echo("    • Dimensions: 1024")
    else:
        click.echo("Use --show to display current configuration")


if __name__ == "__main__":
    cli()