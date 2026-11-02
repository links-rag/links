def main() -> None:
    """Start the retrieval API on localhost port 8000."""
    import uvicorn

    uvicorn.run("arxiv_retrieval_jacky.app:app", host="127.0.0.1", port=8000)
