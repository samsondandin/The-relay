"""Entrypoint script for running The Relay service and utilities."""

import argparse
import sys
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="The Relay - Research Assistant CLI & Runner")
    parser.add_argument(
        "command",
        choices=["serve", "benchmarks", "eval", "simulator"],
        nargs="?",
        default="serve",
        help="Command to run: 'serve' (main app), 'benchmarks' (load suite), 'eval' (15 questions), 'simulator' (stand-alone simulator server)"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--workers", type=int, default=1, help="Worker count")

    args = parser.parse_args()

    if args.command == "serve":
        print(f"Starting The Relay API Server on {args.host}:{args.port}...")
        uvicorn.run("app.main:app", host=args.host, port=args.port, reload=False, workers=args.workers)

    elif args.command == "simulator":
        print(f"Starting standalone Simulator Service on {args.host}:{args.port}...")
        uvicorn.run("simulator_service.main:app", host=args.host, port=args.port, reload=False)

    elif args.command == "benchmarks":
        from benchmarks.run_benchmarks import run_all_benchmarks
        import asyncio
        asyncio.run(run_all_benchmarks())

    elif args.command == "eval":
        from benchmarks.run_benchmarks import run_evaluation_set
        import asyncio
        asyncio.run(run_evaluation_set())


if __name__ == "__main__":
    main()
