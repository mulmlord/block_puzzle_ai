import argparse

import torch
import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch the AI play in your browser.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    torch.set_num_threads(4)
    uvicorn.run("block_puzzle_ai.viewer.server:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
