"""CLI script for pinner2026_jpo_code."""

import typer
from rich.console import Console

app = typer.Typer()
console = Console()


@app.command()
def main():
    """CLI script for pinner2026_jpo_code."""
    console.print(r"""
  $$\      $$\           $$\                                             
  $$ | $\  $$ |          $$ |                                            
  $$ |$$$\ $$ | $$$$$$\  $$ | $$$$$$$\  $$$$$$\  $$$$$$\$$$$\   $$$$$$\  
  $$ $$ $$\$$ |$$  __$$\ $$ |$$  _____|$$  __$$\ $$  _$$  _$$\ $$  __$$\ 
  $$$$  _$$$$ |$$$$$$$$ |$$ |$$ /      $$ /  $$ |$$ / $$ / $$ |$$$$$$$$ |
  $$$  / \$$$ |$$   ____|$$ |$$ |      $$ |  $$ |$$ | $$ | $$ |$$   ____|
  $$  /   \$$ |\$$$$$$$\ $$ |\$$$$$$$\ \$$$$$$  |$$ | $$ | $$ |\$$$$$$$\ 
  \__/     \__| \_______|\__| \_______| \______/ \__| \__| \__| \_______|
                                                                      
    """, style="green")
    # https://patorjk.com/software/taag/
    console.print("Replace this message by putting your code into "
               "pinner2026_jpo_code.cli.main")
    console.print("See Typer documentation at https://typer.tiangolo.com/")


if __name__ == "__main__":
    app()
