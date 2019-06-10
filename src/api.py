# coding: utf8

import json

from scripts.fever_gplsi import pipeline
from athene.utils.config import Config
from fever.api.web_server import fever_web_api


def my_sample_fever(*args):
    # Set up and initialize model

    # A prediction function that is called by the API
    def baseline_predict(instances):
        predictions = []

        with open(Config.raw_test_set(), "w") as fp:
            for instance in instances:
                fp.write("%s\n" % json.dumps(instance))

        pipeline.main()

        with open(Config.submission_file(), "r") as fp:
            for line in fp:
                predictions.append(json.loads(line.strip()))

        return predictions

    return fever_web_api(baseline_predict)


if __name__ == "__main__":
    my_sample_fever().run(host="0.0.0.0", port=5000)
