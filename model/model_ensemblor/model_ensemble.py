import torch
from ensemble_utils import box_torch_ops

def model_ensemble(model_results_list, method_name):
    """
        ensemble different model results to get better results.

        Args:
            model_results_list: list. different model predictions path.
            method_name: str. ensemble method name.
        return:
            different model ensemble results.

    """
    model_results_data = []
    for path in model_results_list:
        with open(path, 'rb') as fo:
            model_results_data.append(pickle.load(fo, encoding='bytes'))

    label = list(model_results_data[0].keys())
    prediction_dicts = []
    final_result = {}

    box_preds = []
    scores = []
    labels = []

    for i in range(len(label)):
        for model_result_data in model_results_data:
            model_result_data_ele = model_result_data[label[i]]
            box_preds.append(model_result_data_ele['box3d_lidar'])
            scores.append(model_result_data_ele['scores'])
            preds.append(model_result_data_ele['label_preds'])
            gfpn_box_data = gfpn_boxes_data[label[i]]

        box_preds = torch.stack(box_preds).to('cuda:0')
        scores = torch.stack(scores).to('cuda:0')
        labels = torch.stack(labels).to('cuda:0')

        if method_name == 'nms':
            boxes_for_nms = box_preds[:, [0, 1, 2, 3, 4, 5, -1]]

            selected = box_torch_ops.rotate_nms_pcdet(boxes_for_nms.float(), scores.float(),
                                                      thresh=0.2,
                                                      pre_maxsize=1000,
                                                      post_max_size=83)

        selected_boxes = box_preds[selected]
        selected_scores = scores[selected]
        selected_labels = labels[selected]

        prediction_dict = {
            'box3d_lidar': boxes,
            'scores': scores,
            'label_preds': labels,
            'metadata': model_results_data[0]['metadata']
        }

        final_result[label[i]] = prediction_dict
        prediction_dicts.append(prediction_dict)

    return prediction_dicts

