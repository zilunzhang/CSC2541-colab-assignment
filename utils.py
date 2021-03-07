import os
import torch
import cv2


def get_sample_image(way_num, support_inputs, query_inputs, support_targets, query_targets, save_dir):
    # only support episode number = 1
    assert support_inputs.shape[0] == query_inputs.shape[0] == support_targets.shape[0] == query_targets.shape[0]
    os.makedirs(save_dir, exist_ok=True)
    # (5, 84, 84, 3)
    support_data_permute = support_inputs.permute(0, 1, 3, 4, 2).squeeze(0)
    # (75, 84, 84, 3)
    query_data_permute = query_inputs.permute(0, 1, 3, 4, 2).squeeze(0)

    support_data_reshape = torch.reshape(support_data_permute, (way_num, -1, *support_data_permute.shape[1:]))
    query_data_reshape = torch.reshape(query_data_permute, (way_num, -1, *query_data_permute.shape[1:]))
    device = support_inputs.get_device()
    # (5, 1+15, 84, 84, 3)
    black = torch.zeros(support_data_reshape.shape[0], 1, *support_data_reshape.shape[-3:]) + 1
    black = black.cuda() if device != -1 else black
    complete_tensor = torch.cat([support_data_reshape, black, query_data_reshape], dim=1)
    present_list = []
    for row in complete_tensor:
        tensor_list = [tensor for tensor in row]
        tensor_row = torch.cat(tensor_list, dim=1)
        present_list.append(tensor_row)
    present_tensor = torch.cat(present_list, dim=0)
    img = present_tensor.cpu().numpy() * 255
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(save_dir, 'sampled_image.png'), img)

