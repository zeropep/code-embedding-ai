import click
import structlog
from dotenv import load_dotenv

load_dotenv()

logger = structlog.get_logger(__name__)


@click.group()
@click.option("--config", default="config/pipeline.yaml", help="Configuration file path")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def cli(config: str, verbose: bool):
    if verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    logger.info("Code Embedding Pipeline CLI", config_file=config)


@cli.command()
@click.option("--repo-path", required=True, help="Path to source code repository")
@click.option("--output-db", default="./chroma_db", help="Output ChromaDB path")
def process(repo_path: str, output_db: str):
    click.echo(f"Processing repository: {repo_path}")
    click.echo(f"Output database: {output_db}")


@cli.command()
@click.option("--repo-path", required=True, help="Path to source code repository")
@click.option("--output-db", default="./chroma_db", help="ChromaDB path")
def update(repo_path: str, output_db: str):
    click.echo(f"Updating embeddings for: {repo_path}")


@cli.command()
@click.option("--query", required=True, help="Search query")
@click.option("--db-path", default="./chroma_db", help="ChromaDB path")
@click.option("--limit", default=10, help="Number of results")
def search(query: str, db_path: str, limit: int):
    click.echo(f"Searching for: {query}")


@cli.command()
def serve():
    click.echo("Starting API server...")


if __name__ == "__main__":
    cli()
