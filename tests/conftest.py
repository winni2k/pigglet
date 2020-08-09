import pytest
import ray


@pytest.fixture(scope="module")
def ray_init():
    ray.init()
