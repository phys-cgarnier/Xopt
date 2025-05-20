import json
from copy import deepcopy

import pytest
import pandas as pd
import numpy as np

from xopt.errors import VOCSError
from xopt.generator import Generator
from xopt.generators import (
    get_generator,
    get_generator_defaults,
    list_available_generators,
)
from xopt.resources.testing import TEST_VOCS_BASE


class PatchGenerator(Generator):
    """
    Test generator class for testing purposes.
    """

    name = "test_generator"
    supports_batch_generation: bool = True
    supports_single_objective: bool = True
    supports_constraints: bool = True

    def generate(self, n_candidates) -> list[dict]:
        pass


class TestGenerator:
    def test_init(self):
        PatchGenerator(vocs=TEST_VOCS_BASE)

        test_vocs = deepcopy(TEST_VOCS_BASE)
        test_vocs.objectives.update({"y2": "MINIMIZE"})

        with pytest.raises(VOCSError):
            PatchGenerator(vocs=test_vocs)

    def test_add_data(self):
        test_vocs = deepcopy(TEST_VOCS_BASE)
        gen = PatchGenerator(vocs=test_vocs)

        gen.add_data(pd.DataFrame({"x1": 1, "x2": 2, "y1": 3}, index=[0]))

        assert gen.data.shape == (1, 3)

        # add a large amount of data
        data = pd.DataFrame(
            {
                "x1": np.random.rand(100),
                "x2": np.random.rand(100),
                "y1": np.random.rand(100),
            }
        )

        gen.add_data(data)
        assert gen.data.shape == (101, 3)

        # make sure that the inidices are correct
        assert gen.data.index.tolist() == list(range(101))

    def test_data_index_is_int(self):
        test_vocs = deepcopy(TEST_VOCS_BASE)
        gen = PatchGenerator(vocs=test_vocs)

        # Add data with string index
        df = pd.DataFrame({"x1": [1], "x2": [2], "y1": [3]}, index=["0"])
        gen.add_data(df)
        assert gen.data.index.dtype == int

        # Add more data with mixed index types
        df2 = pd.DataFrame({"x1": [4, 5], "x2": [6, 7], "y1": [8, 9]}, index=["1", 2])
        gen.add_data(df2)
        assert gen.data.index.dtype == int
        assert gen.data.index.tolist() == list(range(3))

        # Add data with integer index
        df3 = pd.DataFrame({"x1": [10], "x2": [11], "y1": [12]}, index=[3])
        gen.add_data(df3)
        assert gen.data.index.dtype == int
        assert gen.data.index.tolist() == list(range(4))

    @pytest.mark.parametrize("name", list_available_generators())
    def test_serialization_loading(self, name):
        gen_config = get_generator_defaults(name)
        gen_class = get_generator(name)

        if name in ["mobo", "cnsga", "mggpo"]:
            test_vocs = deepcopy(TEST_VOCS_BASE)
            test_vocs.objectives.update({"y2": "MINIMIZE"})
            gen_config["reference_point"] = {"y1": 10.0, "y2": 1.5}
            json.dumps(gen_config)

            gen_class(vocs=test_vocs, **gen_config)
        elif name in ["nsga2"]:
            test_vocs = deepcopy(TEST_VOCS_BASE)
            test_vocs.objectives.update({"y2": "MINIMIZE"})
            json.dumps(gen_config)

            gen_class(vocs=test_vocs, **gen_config)

        elif name in ["multi_fidelity"]:
            test_vocs = deepcopy(TEST_VOCS_BASE)
            test_vocs.constraints = {}
            json.dumps(gen_config)

            gen_class(vocs=test_vocs, **gen_config)
        elif name in ["bayesian_exploration"]:
            test_vocs = deepcopy(TEST_VOCS_BASE)
            test_vocs.objectives = {}
            test_vocs.observables = ["f"]
            json.dumps(gen_config)

            gen_class(vocs=test_vocs, **gen_config)
        else:
            test_vocs = deepcopy(TEST_VOCS_BASE)
            test_vocs.constraints = {}
            json.dumps(gen_config)

            gen_class(vocs=test_vocs, **gen_config)
