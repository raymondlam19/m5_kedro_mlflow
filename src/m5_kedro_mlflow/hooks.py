# Copyright 2020 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
# or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Project hooks."""
import cProfile
import io
import pstats
from typing import Any, Dict, Iterable, Optional
from unittest import mock

from kedro.config import ConfigLoader
from kedro.framework.hooks import hook_impl
from kedro.io import DataCatalog
from kedro.pipeline.node import Node
from kedro.pipeline.pipeline import Pipeline

import yaml
import os
from datetime import datetime
import numpy as np


class ProjectHooks:
    @hook_impl
    def register_config_loader(
        self, conf_paths: Iterable[str], env: str, extra_params: Dict[str, Any]
    ) -> ConfigLoader:
        return ConfigLoader(conf_paths)

    @hook_impl
    def register_catalog(
        self,
        catalog: Optional[Dict[str, Dict[str, Any]]],
        credentials: Dict[str, Dict[str, Any]],
        load_versions: Dict[str, str],
        save_version: str,
    ) -> DataCatalog:
        return DataCatalog.from_config(
            catalog, credentials, load_versions, save_version
        )


# class ProfilerHooks:
#     def __init__(self):
#         self.profiler = None

#     @hook_impl
#     def before_node_run(self, node: Node):
#         if node.name == "lgbm_train":
#             self.profiler = cProfile.Profile()
#             self.profiler.enable()
#             print("Start profilling")

#     @hook_impl
#     def after_node_run(self, node: Node):
#         if node.name == "lgbm_train":
#             self.profiler.disable()
#             stats = pstats.Stats(self.profiler, stream=io.StringIO()).sort_stats(
#                 "cumtime"
#             )
#             stats.strip_dirs().print_stats()
#             stats.dump_stats("logs\\profiles.prof")
#             self.profiler = None


# class HyperOptHooks:
#     def __init__(self):
#         """_summary_
#         This hook tries to integrate ml_trains + hyperopt with optuna-dashboard, for visualization of hyper-parameters tunning
#         """
#         self.study = None
#         self.target = None
#         self.strategy = None
#         self.dist = {}

#     @hook_impl
#     def before_pipeline_run(self, run_params: Dict[str, Any], pipeline: Pipeline):
#         if "_hyperopt" not in run_params["pipeline_name"]:
#             return

#         def get_config(config_path, section):
#             with open(config_path, "r") as f:
#                 _config = yaml.load(f, yaml.FullLoader)
#             return _config[section]

#         config_path = os.path.join(CDF_HOME, "cdf-yoe", "run", "lgbm_hyperopt.yml")
#         hyperopt_params = get_config(config_path, "hyperopt")
#         if self.study is None:
#             self.target = hyperopt_params["target"]["name"]
#             self.strategy = (
#                 "minimize"
#                 if hyperopt_params["target"]["strategy"] == "min"
#                 else "maximize"
#             )
#             self.study = optuna.create_study(
#                 storage="sqlite:///optuna.sqlite3",
#                 study_name=hyperopt_params["study_name"] + str(datetime.now()),
#                 direction=self.strategy,
#             )

#         hyperopt_model_params_dict = hyperopt_params["model"]["lgbm"]
#         for param, args in hyperopt_model_params_dict.items():
#             if args["method"] == "choice" and "values" in args.keys():
#                 self.dist.update({param: CategoricalDistribution(args["values"])})
#             elif args["method"] == "choice":
#                 self.dist.update(
#                     {param: IntUniformDistribution(args["low"], args["high"])}
#                 )
#             elif args["method"] == "uniform":
#                 self.dist.update(
#                     {param: UniformDistribution(args["low"], args["high"])}
#                 )
#             elif args["method"] == "loguniform":
#                 self.dist.update(
#                     {
#                         param: LogUniformDistribution(
#                             np.exp(args["low"]), np.exp(args["high"])
#                         )
#                     }
#                 )
#             elif args["method"] == "quniform":
#                 self.dist.update(
#                     {
#                         param: DiscreteUniformDistribution(
#                             args["low"], args["high"], args["q"]
#                         )
#                     }
#                 )
#             else:
#                 raise NotImplementedError(
#                     f"method={args['method']} of hyperopt is not compatable with optuna-dashboard"
#                 )

#     @hook_impl
#     def after_node_run(
#         self, node: Node, inputs: Dict[str, Any], outputs: Dict[str, Any]
#     ):
#         if not (
#             node.name == "train_base_hyperopt" or node.name == "train_svvd_hyperopt"
#         ):
#             return
#         params = {}
#         model_params = outputs[list(outputs.keys())[0]].params
#         for key in self.dist.keys():
#             params.update({key: model_params[key]})

#         trial = optuna.trial.create_trial(
#             params=params,
#             distributions=self.dist,
#             value=outputs[self.target],
#             user_attrs={"clearml_exp_id": inputs["clearml_exp_id"]},
#         )
#         self.study.add_trial(trial)


project_hooks = ProjectHooks()
