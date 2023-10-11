import logging
import asyncio

# from modules.processor import Processor
from modules.processor import Processor

logging.basicConfig(level=logging.INFO)


async def main():
    processor = Processor()
    await processor.run()


if __name__ == "__main__":
    # main()
    asyncio.run(main())
