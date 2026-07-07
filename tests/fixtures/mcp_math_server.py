"""A tiny MCP server used by the MCP integration tests (stdio transport)."""

from mcp.server.fastmcp import FastMCP

server = FastMCP("math")


@server.tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@server.tool()
def shout(text: str) -> str:
    """Uppercase some text."""
    return text.upper()


if __name__ == "__main__":
    server.run()  # stdio
