from pathlib import Path

from acg.cli import create_clients
from acg.example import build_example_task
from acg.runtime.agent import Agent

chat, embedding = create_clients({})
try:
    result = Agent(chat, embedding).run(
        build_example_task(), output_dir=Path("outputs/example"), save_trace=True
    )
    print(result.status, result.answer)
finally:
    chat.close()
    embedding.client.close()
