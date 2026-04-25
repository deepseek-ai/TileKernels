import hashlib
import random

import pytest
import torch


def pytest_addoption(parser):
    parser.addoption('--seed', type=int, default=0)

@pytest.fixture(autouse=True)
def seed(request):
    base = request.config.getoption('--seed')
    node_hash = int(hashlib.sha256(
        request.node.nodeid.encode()
    ).hexdigest(), 16) % (2**31)
    seed = base + node_hash
    torch.manual_seed(seed)
    random.seed(seed)
    return seed
