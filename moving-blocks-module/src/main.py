import asyncio

# Importing registers the Generic service model with the module registry.
from models.moving_blocks import MovingBlocks  # noqa: F401
from viam.module.module import Module


if __name__ == "__main__":
    asyncio.run(Module.run_from_registry())
