"""Console script for synthetic_sampling."""
import synthetic_sampling

import typer
from rich.console import Console

app = typer.Typer()
console = Console()


@app.command()
def main():
    """Console script for synthetic_sampling."""
    console.print("Replace this message by putting your code into "
               "synthetic_sampling.cli.main")
    console.print("See Typer documentation at https://typer.tiangolo.com/")
    


if __name__ == "__main__":
    app()
