import random
import json
import evaluate as hfeval

def evaluate(test_annotation_file, user_submission_file, phase_codename, **kwargs):
    print("Starting Evaluation.....")
    """
    Evaluates the submission for a particular challenge phase and returns score
    Arguments:

        `test_annotations_file`: Path to test_annotation_file on the server
        `user_submission_file`: Path to file submitted by the user
        `phase_codename`: Phase to which submission is made

        `**kwargs`: keyword arguments that contains additional submission
        metadata that challenge hosts can use to send slack notification.
        You can access the submission metadata
        with kwargs['submission_metadata']

        Example: A sample submission metadata can be accessed like this:
        >>> print(kwargs['submission_metadata'])
        {
            'status': u'running',
            'when_made_public': None,
            'participant_team': 5,
            'input_file': 'https://abc.xyz/path/to/submission/file.json',
            'execution_time': u'123',
            'publication_url': u'ABC',
            'challenge_phase': 1,
            'created_by': u'ABC',
            'stdout_file': 'https://abc.xyz/path/to/stdout/file.json',
            'method_name': u'Test',
            'stderr_file': 'https://abc.xyz/path/to/stderr/file.json',
            'participant_team_name': u'Test Team',
            'project_url': u'http://foo.bar',
            'method_description': u'ABC',
            'is_public': False,
            'submission_result_file': 'https://abc.xyz/path/result/file.json',
            'id': 123,
            'submitted_at': u'2017-03-20T19:22:03.880652Z'
        }
    """

    metrics_list = ['bleu','meteor','rouge']
    metrics = {m:hfeval.load(m) for m in metrics_list}


    with open(user_submission_file, "r") as test_file_object:
        test_data = json.load(test_file_object)
    with open(test_annotation_file, "r") as ground_truth_file_object:
        ground_truth_data = json.load(ground_truth_file_object)

    test_dict = {item: test_data[item][0] for item in test_data}
    ground_truth_dict = {item: test_data[item][0] for item in ground_truth_data}
    
    gts, preds = [], []
    for key in ground_truth_dict.keys():
        gts.append(ground_truth_dict[key])
        if key in test_dict:
            preds.append(test_dict[key])
        else:
            preds.append('')
    
    bleu_score = metrics['bleu'].compute(predictions=preds,references=gts)
    meteor_score = metrics['meteor'].compute(predictions=preds,references=gts)
    rouge_score = metrics['rouge'].compute(predictions=preds,references=gts)

    # for id, test_emotion in test_dict.items():
    #     ground_truth_emotion = ground_truth_dict.get(id)
    #     if ground_truth_emotion is not None:
    #         bleu_score = metrics['bleu'].compute(predictions=test_emotion,references=ground_truth_emotion)
    #         meteor_score = metrics['meteor'].compute(predictions=test_emotion,references=ground_truth_emotion)
    #         rouge_score = metrics['rouge'].compute(predictions=test_emotion,references=ground_truth_emotion)

    output = {}
    if phase_codename == "dev":
        print("Evaluating for Dev Phase")
        output["result"] = [
            {
                "train_split": {
                    "BLEU": bleu_score['bleu'],
                    "METEOR": meteor_score['meteor'],
                    "ROUGE1": rouge_score['rouge1'],
                    "ROUGE2": rouge_score['rouge2'],
                    "ROUGEL": rouge_score['rougeL'],
                }
            }
        ]
        # To display the results in the result file
        output["submission_result"] = output["result"][0]["train_split"]
        print("Completed evaluation for Dev Phase")
    elif phase_codename == "test":
        print("Evaluating for Test Phase")
        output["result"] = [
            {
                "train_split": {
                    "BLEU": bleu_score['bleu'],
                    "METEOR": meteor_score['meteor'],
                    "ROUGE1": rouge_score['rouge1'],
                    "ROUGE2": rouge_score['rouge2'],
                    "ROUGEL": rouge_score['rougeL'],
                }
            },
            {
                "test_split": {
                    "BLEU": bleu_score['bleu'],
                    "METEOR": meteor_score['meteor'],
                    "ROUGE1": rouge_score['rouge1'],
                    "ROUGE2": rouge_score['rouge2'],
                    "ROUGEL": rouge_score['rougeL'],                    
                }
            },
        ]
        # To display the results in the result file
        output["submission_result"] = output["result"][0]
        print("Completed evaluation for Test Phase")
    return output

