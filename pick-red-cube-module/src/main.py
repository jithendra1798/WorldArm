import asyncio

# Importing the model registers it with EasyResource / the global registry.
from models.picker import PickRedCube  # noqa: F401
from viam.module.module import Module


if __name__ == "__main__":
    asyncio.run(Module.run_from_registry())
