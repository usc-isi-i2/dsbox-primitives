import json
import os
import subprocess
import d3m
import pandas as pd
import shutil
import traceback
import gzip

from template import DATASET_MAPPER
from library import *  # import testing template
from dsbox.datapreprocessing.cleaner import config as cleaner_config

TEMPLATE_LIST = []
# add templates here
TEMPLATE_LIST.append(DefaultClassificationTemplate())
TEMPLATE_LIST.append(DefaultTimeseriesCollectionTemplate())
TEMPLATE_LIST.append(DefaultRegressionTemplate())
TEMPLATE_LIST.append(VotingTemplate())
TEMPLATE_LIST.append(TA1ImageProcessingRegressionTemplate())
# ends

def execute_shell_code(shell_command):
    print("excuting...")
    print(shell_command)
    p = subprocess.Popen(shell_command, shell=True, stdout=subprocess.PIPE, universal_newlines=True)
    p.wait()
    print("excuting finished!")


def get_meta_json(dataset_name):
    # generate the meta file for pipelines
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


def get_primitive_hitted(config):
    """
        Return a list of DSBOX primitives that are found in the config file
        We should only add sample pipelines for our own primitives
    """
    primitive_hitted = []
    for each_primitive in config.values():
        temp = each_primitive['primitive']
        if temp[-5:] == "DSBOX":
            primitive_hitted.append(temp)
    return primitive_hitted


def generate_pipelines(template, config: dict, meta_json):
    """
        Generate sample pipelines and corresponding meta
    """
    primitive_hitted = get_primitive_hitted(config)
    for each_primitive in primitive_hitted:
        output_dir = os.path.join("output", 'v' + cleaner_config.D3M_API_VERSION,
                              cleaner_config.D3M_PERFORMER_TEAM, each_primitive,
                              cleaner_config.VERSION)
        output_pipeline_dir = os.path.join(output_dir, "pipelines")
        output_pipeline_runs_dir = os.path.join(output_dir, "pipeline_runs")
        os.makedirs(output_pipeline_dir, exist_ok=True)
        os.makedirs(output_pipeline_runs_dir, exist_ok=True)
        failed = []
        try:
            # generate the new pipeline
            # updated for d3m v2019.11.10: .meta file not needed, an extra file pipeline_runs needed
            pipeline = template.to_pipeline(config)
            pipeline_json = pipeline.to_json_structure()
            print("Generating at " + output_pipeline_dir +  "/" + pipeline_json['id'] + "...")
            file_name = os.path.join(output_pipeline_dir, pipeline_json['id']+".json")
            
            with open(file_name, "w") as f:
                json.dump(pipeline_json, f, separators=(',', ':'),indent=4)

            # copy pipeline_run files
            file_count = len(os.listdir(output_pipeline_runs_dir))
            pipeline_runs_file = os.path.join(output_pipeline_runs_dir, "pipeline_run_{}.yaml.gzip".format(str(file_count + 1)))
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
            failed.append(file_name)
            print("!!!!!!!")
            print("failed!")
            print("!!!!!!!")
            traceback.print_exc()

    return failed


def remove_temp_files_generate_pipeline_runs():
    tmp_files = os.listdir("tmp")
    for each_file in tmp_files:
        file_path = os.path.join("tmp", each_file)
        # if each_file != "pipeline_runs.yaml":
        os.remove(file_path)


def prepare_for_runtime():
    """
    generate necessary files for runtime
    """
    # score pipeline
    print("*" * 100)
    print("preparing for running now...")
    score_pipeline_path = os.path.join(str(d3m.__path__[0]), "contrib/pipelines/f596cd77-25f8-4d4c-a350-bb30ab1e58f6.yml")
    d3m_runtime_command = "python3 -m d3m pipeline describe --not-standard-pipeline " + \
    score_pipeline_path + " > normalizedScoringPipeline.json"
    execute_shell_code(d3m_runtime_command)
    if os.path.exists("normalizedScoringPipeline.json"):
        print("generate scoring pipeline succeeded.")
    else:
        raise ValueError("Scoring pipeline not find!")

    # corex primitive.json
    import corex_text
    corex_package_path = os.path.abspath(os.path.join(os.path.dirname(corex_text.__file__ ), 'generate_primitive_json.py'))
    generate_corex_primitive_json_files = "python3 " + corex_package_path + " dsbox-unit-test-datasets"
    execute_shell_code(generate_corex_primitive_json_files)
    print("Generate corex related primitive.json finished!")
    print("*" * 100)
    print("preparing finished!")

def test_pipeline(each_template, config, test_dataset_id):
    try:
        pipeline = each_template.to_pipeline(config)
        pipeline_json = pipeline.to_json_structure()
        os.makedirs("tmp", exist_ok=True)
        temp_pipeline = os.path.join("tmp/test_pipeline.json")
        with open(temp_pipeline, "w") as f:
            json.dump(pipeline_json, f)

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
        volume_dir = os.getcwd(),
        problem_doc = "dsbox-unit-test-datasets/" + test_dataset_id + "/TRAIN/problem_TRAIN/problemDoc.json",
        train_dataset_doc = "dsbox-unit-test-datasets/" + test_dataset_id + "/TRAIN/dataset_TRAIN/datasetDoc.json",
        test_dataset_doc = "dsbox-unit-test-datasets/" + test_dataset_id + "/TEST/dataset_TEST/datasetDoc.json",
        score_dataset_doc = score_dataset_doc,
        pipeline_runs_yaml_doc = "tmp/pipeline_runs.yaml",
        prediction_csv = "tmp/predictions.csv",
        score_csv = "tmp/score.csv",
        )
        execute_shell_code(d3m_runtime_command)
        try:
            # check score file
            predictions = pd.read_csv("tmp/score.csv")
        except Exception:
            print("predictions file load failed, please check the pipeline.")
            return False

        # if everything OK
        return True
    except Exception:
        raise ValueError("Running train-test with config" + each_template + "failed!")
        return False


def main():
    # clean up the old output files if necessary
    try:
        shutil.rmtree("output")
    except Exception:
        pass

    # config_list = os.listdir("pipeline_configs")
    # config_list = list(map(lambda x: x.generate_pipeline_direct().config, TEMPLATE_LIST))
    # generate pipelines for each configuration
    prepare_for_runtime()
    for each_template in TEMPLATE_LIST:
        print("*" * 100)
        print("Now processing template", str(each_template))
        config = each_template.generate_pipeline_direct().config
        datasetID = DATASET_MAPPER[each_template.template['runType'].lower()]
        meta_json = get_meta_json(datasetID)
        result = test_pipeline(each_template, config, datasetID)
        # only generate the pipelines with it pass the test
        if result:
            predictions = pd.read_csv("tmp/score.csv")
            print("*"*100)
            print("unit test pipeline's score for " + datasetID)
            print(predictions)
            print("*"*100)
            print("Test pipeline passed! Now generating the pipeline json files...")
            failed = generate_pipelines(each_template, config, meta_json)
            remove_temp_files_generate_pipeline_runs()
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

    

if __name__ == "__main__":
    main()
