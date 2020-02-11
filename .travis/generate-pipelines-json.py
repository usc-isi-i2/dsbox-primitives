''
import gzip
import json
import os
import shutil
import subprocess
import traceback
import typing

from collections import defaultdict

import pandas as pd
import d3m
from dsbox_corex import corextext

from template import DATASET_MAPPER
from template import DSBoxTemplate
from library import *  # import testing template
from dsbox.datapreprocessing.cleaner import config as cleaner_config

TEMPLATE_LIST = []

# add templates here
# TEMPLATE_LIST.append(UU3TestTemplate()) # no enough memory for travis ci
TEMPLATE_LIST.append(DefaultObjectDetectionTemplate())
TEMPLATE_LIST.append(ARIMATemplate())
TEMPLATE_LIST.append(TA1ImageProcessingRegressionTemplate())
TEMPLATE_LIST.append(TA1ImageProcessingRegressionTemplate2())
TEMPLATE_LIST.append(DefaultClassificationTemplate())
TEMPLATE_LIST.append(DefaultClassificationTemplate2())
TEMPLATE_LIST.append(DefaultTimeseriesCollectionTemplate())
TEMPLATE_LIST.append(DefaultRegressionTemplate())
TEMPLATE_LIST.append(DefaultRegressionTemplate2())
TEMPLATE_LIST.append(VotingTemplate())
TEMPLATE_LIST.append(HorizontalVotingTemplate())
# ends


class DsboxPrimitiveUnitTest:
    def __init__(self, templates_to_test: typing.List[DSBoxTemplate], include_corex_primitives: bool=True):
        self.templates_to_test = templates_to_test
        self.prepare_for_runtime()
        self.include_corex_primitives = include_corex_primitives
        self.all_primitives_hit_count = defaultdict(int)
        self.corex_primitives = []
        self.template_to_pipeline_ids = {}

    @staticmethod
    def execute_shell_code(shell_command: str):
        print("excuting...")
        print(shell_command)
        p = subprocess.Popen(shell_command, shell=True, stdout=subprocess.PIPE, universal_newlines=True)
        p.wait()
        print("excute finished!")

    # generate the meta file for pipelines, should not be used after d3m v2019.11.10
    @staticmethod
    def get_meta_json(dataset_name):
        # if pipeline_type == "classification":
        #     dataset_name = "38_sick"
        # elif pipeline_type == "regression":
        #     dataset_name = "196_autoMpg"
        # # more pipeline types needed

        meta_json = {
                    "problem": dataset_name + "_problem",
                    "full_inputs": [dataset_name + "_dataset"],
                    "train_inputs": [dataset_name + "_dataset_TRAIN"],
                    "test_inputs": [dataset_name + "_dataset_TEST"],
                    "score_inputs": [dataset_name + "_dataset_SCORE"]
                }

        return meta_json

    @staticmethod
    def remove_temp_files_generate_pipeline_runs():
        tmp_files = os.listdir("tmp")
        for each_file in tmp_files:
            file_path = os.path.join("tmp", each_file)
            # if each_file != "pipeline_runs.yaml":
            os.remove(file_path)

    def prepare_for_runtime(self):
        """
        generate necessary files for runtime
        """
        # clean up the old output files if necessary
        try:
            shutil.rmtree("output")
        except Exception:
            pass
        # score pipeline
        print("*" * 100)
        print("preparing for running now...")
        score_pipeline_path = os.path.join(str(d3m.__path__[0]), "contrib/pipelines/f596cd77-25f8-4d4c-a350-bb30ab1e58f6.yml")
        d3m_runtime_command = "python3 -m d3m pipeline describe --not-standard-pipeline " + \
        score_pipeline_path + " > normalizedScoringPipeline.json"
        self.execute_shell_code(d3m_runtime_command)
        if os.path.exists("normalizedScoringPipeline.json"):
            print("generate scoring pipeline succeeded.")
        else:
            raise ValueError("Scoring pipeline not find!")

        # genereate corex primitive.json
        corex_package_path = os.path.abspath(os.path.join(os.path.dirname(corextext.__path__._path[0]),"..", 'generate_primitive_json.py'))
        generate_corex_primitive_json_files = "python3 " + corex_package_path + " output"
        self.execute_shell_code(generate_corex_primitive_json_files)
        corex_primitives_json_loc = os.path.join("output", "v" + cleaner_config.D3M_API_VERSION,
                                                 cleaner_config.D3M_PERFORMER_TEAM)
        corex_primitives_generate_failed = True
        if os.path.exists(corex_primitives_json_loc):
            self.corex_primitives = os.listdir(corex_primitives_json_loc)
            if len(self.corex_primitives) != 0:
                corex_primitives_generate_failed = True

        print("*" * 100)
        if corex_primitives_generate_failed:
            print("[WARNING]: corex primitives generate failed!")
        else:
            print("following corex primitives found:")
            print(str(self.corex_primitives))
        print("preparing finished!")

    def get_primitive_hitted(self, config):
        """
            Return a list of DSBOX primitives that are found in the config file
            We should only add sample pipelines for our own primitives.

            notice: for corex primitives, due to the reason that they are in
            different branch from ours, it has to be uploaded and merged to
            d3m `primitives` branch first, and then we can submit the sample
            pipelines, otherwise the CI on d3m will failed.
        """
        primitive_hitted = []
        for each_primitive in config.values():
            temp = each_primitive['primitive']
            if temp[-5:] == "DSBOX":
                if temp in self.corex_primitives and not self.include_corex_primitives:
                    continue
                primitive_hitted.append(temp)
                self.all_primitives_hit_count[temp] += 1
        return primitive_hitted

    def generate_pipelines(self, each_template, config: dict):
        """
            Generate sample pipelines and corresponding meta
        """
        primitive_hitted = self.get_primitive_hitted(config)
        failed = []
        for each_primitive in primitive_hitted:
            if each_primitive in self.corex_primitives:
                output_dir = os.path.join("output", 'v' + cleaner_config.D3M_API_VERSION,
                                  cleaner_config.D3M_PERFORMER_TEAM, each_primitive,
                                  corex_text.cfg_.VERSION)
            else:
                output_dir = os.path.join("output", 'v' + cleaner_config.D3M_API_VERSION,
                                  cleaner_config.D3M_PERFORMER_TEAM, each_primitive,
                                  cleaner_config.VERSION)

            output_pipeline_dir = os.path.join(output_dir, "pipelines")
            output_pipeline_runs_dir = os.path.join(output_dir, "pipeline_runs")
            os.makedirs(output_pipeline_dir, exist_ok=True)
            os.makedirs(output_pipeline_runs_dir, exist_ok=True)
            try:
                # generate the new pipeline
                # updated for d3m v2019.11.10: .meta file not needed, an extra file pipeline_runs needed

                # copy pipeline files
                temp_location = "tmp/test_pipeline.json"
                pipeline_id = self.template_to_pipeline_ids[str(each_template)]
                new_location = os.path.join(output_pipeline_dir, pipeline_id + ".json")
                shutil.copy(temp_location, new_location)

                # copy pipeline_run files
                file_count = len(os.listdir(output_pipeline_runs_dir))
                pipeline_runs_file = os.path.join(output_pipeline_runs_dir, "pipeline_run_{}.yaml.gz".format(str(file_count + 1)))
                with open("tmp/pipeline_runs.yaml", "rb") as f:
                    data = f.read()
                bindata = bytearray(data)
                with gzip.open(pipeline_runs_file, "wb") as f:
                    f.write(bindata)
                # meta_name = os.path.join(outdir, pipeline_json['id']+".meta")
                # with open(meta_name, "w") as f:
                #     json.dump(meta_json, f, separators=(',', ':'),indent=4)
                print("succeeded!")

            except Exception as e:
                failed.append(each_primitive)
                print("!!!!!!!")
                print("failed!")
                print("!!!!!!!")
                traceback.print_exc()

        return failed

    def test_pipeline(self, each_template, config, test_dataset_id):
        try:
            pipeline = each_template.to_pipeline(config)
            pipeline_json = pipeline.to_json_structure()
            self.template_to_pipeline_ids[str(each_template)] = pipeline_json['id']
            os.makedirs("tmp", exist_ok=True)
            temp_pipeline = os.path.join("tmp/test_pipeline.json")
            with open(temp_pipeline, "w") as f:
                json.dump(pipeline_json, f, separators=(',', ':'),indent=4)

            # for some condition, score part may have either `*_SCORE` or `*_TEST`
            score_dataset_doc = "dsbox-unit-test-datasets/" + test_dataset_id + "/SCORE/dataset_SCORE/datasetDoc.json"
            if not os.path.exists(score_dataset_doc):
                score_dataset_doc = "dsbox-unit-test-datasets/" + test_dataset_id + "/SCORE/dataset_TEST/datasetDoc.json"

            d3m_runtime_command = """
            python3 -m d3m runtime \
              --volumes {volume_dir} \
              fit-score \
              --scoring-pipeline normalizedScoringPipeline.json \
              --pipeline tmp/test_pipeline.json \
              --problem {problem_doc} \
              --input {train_dataset_doc} \
              --test-input {test_dataset_doc} \
              --score-input {score_dataset_doc} \
              --output-run {pipeline_runs_yaml_doc} \
              --output {prediction_csv} \
              --scores {score_csv}""".format(
            volume_dir = os.path.abspath(os.path.join(os.getcwd(), "..")),
            problem_doc = "dsbox-unit-test-datasets/" + test_dataset_id + "/TRAIN/problem_TRAIN/problemDoc.json",
            train_dataset_doc = "dsbox-unit-test-datasets/" + test_dataset_id + "/TRAIN/dataset_TRAIN/datasetDoc.json",
            test_dataset_doc = "dsbox-unit-test-datasets/" + test_dataset_id + "/TEST/dataset_TEST/datasetDoc.json",
            score_dataset_doc = score_dataset_doc,
            pipeline_runs_yaml_doc = "tmp/pipeline_runs.yaml",
            prediction_csv = "tmp/predictions.csv",
            score_csv = "tmp/score.csv",
            )
            self.execute_shell_code(d3m_runtime_command)
            try:
                # check score file
                predictions = pd.read_csv("tmp/score.csv")
                print("*"*100)
                print("unit test pipeline's score for {} {}".format(str(each_template), str(test_dataset_id)))
                print(predictions)
                print("*"*100)
            except Exception:
                print("predictions file load failed, please check the pipeline.")
                return False

            # if everything OK
            return True

        except Exception:
            raise ValueError("Running train-test with config " + str(each_template) + " failed!")
            return False

    def start_test(self):
        """
            main entry function for running unit test
        """
        for each_template in self.templates_to_test:
            print("*" * 100)
            print("Now processing template", str(each_template))
            config = each_template.generate_pipeline_direct().config
            datasetID = DATASET_MAPPER[each_template.template['runType'].lower()]
            # meta_json = get_meta_json(datasetID)
            result = self.test_pipeline(each_template, config, datasetID)
            # only generate the pipelines with it pass the test
            if result:
                print("Test pipeline passed! Now generating the pipeline json files...")
                failed = self.generate_pipelines(each_template, config)
                self.remove_temp_files_generate_pipeline_runs()
            else:
                print("Test pipeline not passed! Please check the detail errors")
                raise ValueError("Auto generating pipelines failed")

            if len(failed) != 0:
                print("*"*100)
                print("*"*100)
                print("following primitive pipelines generate failed:")
                for each in failed:
                    print(each)
                return 1

def copy_one_pre_ran_pipeline(in_folder: str) -> None:
    pp_path = None
    pp_run_path = None
    for each_ in os.listdir(in_folder):
        # this is the pipeline.json file
        if each_.endswith("json"):
            pp_path = os.path.join(in_folder, each_)
        elif each_.endswith("yaml.gz"):
            pp_run_path = os.path.join(in_folder, each_)
    
    if pp_run_path is None or pp_path is None:
        raise ValueError("Failed to find pipeline or pipeline run files on folder {}".format(in_folder))
    print("Found pipeline file at {}".format(pp_path))
    print("Found pipeline-run file at {}".format(pp_run_path))

    with open(pp_path, "r") as f:
        pp_read = json.load(f)
    pipeline_id = pp_read['id']
    primitive_hitted = set()
    for each_step in pp_read['steps']:
        primitive_name = each_step['primitive']['python_path']
        if primitive_name.endswith("DSBOX"):
            primitive_hitted.add(primitive_name)

    for each_primitive in primitive_hitted:
        # if each_primitive in self.corex_primitives:
        #     output_dir = os.path.join("output", 'v' + cleaner_config.D3M_API_VERSION,
        #                       cleaner_config.D3M_PERFORMER_TEAM, each_primitive,
        #                       corex_text.cfg_.VERSION)
        # else:
        output_dir = os.path.join("output", 'v' + cleaner_config.D3M_API_VERSION,
                          cleaner_config.D3M_PERFORMER_TEAM, each_primitive,
                          cleaner_config.VERSION)

        output_pipeline_dir = os.path.join(output_dir, "pipelines")
        output_pipeline_runs_dir = os.path.join(output_dir, "pipeline_runs")
        os.makedirs(output_pipeline_dir, exist_ok=True)
        os.makedirs(output_pipeline_runs_dir, exist_ok=True)
        new_location = os.path.join(output_pipeline_dir, pipeline_id + ".json")
        shutil.copy(pp_path, new_location)
        # copy pipeline_run files
        file_count = len(os.listdir(output_pipeline_runs_dir))
        pipeline_runs_file = os.path.join(output_pipeline_runs_dir, "pipeline_run_{}.yaml.gz".format(str(file_count + 1)))
        shutil.copy(pp_run_path, pipeline_runs_file)
        # with open(pp_run_path, "rb") as f:
        #     data = f.read()
        # bindata = bytearray(data)
        # with gzip.open(pipeline_runs_file, "wb") as f:
        #     f.write(bindata)

def copy_pre_ran_pipelines():
    """
        Generate sample pipelines and corresponding meta
    """
    failed = []
    print("*" * 100)
    print("Start copying pre ran pipelines")
    for each_pre_pran_pp_folder in os.listdir("pre_ran_pipelines"):
        if each_pre_pran_pp_folder != "not_used":
            full_path = os.path.join(os.getcwd(), "pre_ran_pipelines" ,each_pre_pran_pp_folder)
            if os.path.isdir(full_path):
                print("Searching on {}".format(full_path))
                try:
                    copy_one_pre_ran_pipeline(full_path)
                    print("succeeded!")
                except Exception as e:
                    failed.append(full_path)
                    print("!!!!!!!")
                    print("failed!")
                    print("!!!!!!!")
                    traceback.print_exc()
    return failed



def main():
    test_unit = DsboxPrimitiveUnitTest(TEMPLATE_LIST)
    test_unit.start_test()
    copy_pre_ran_pipelines()

if __name__ == "__main__":
    main()
