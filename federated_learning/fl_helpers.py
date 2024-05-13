import torch
def compare_models(model1, model2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model1.state_dict().items(), model2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            print(f'Params in layer {key_item_1[0]} are equal')
        else:
            print(f'Params in layer {key_item_1[0]} are different')
            models_differ += 1
    if models_differ == 0:
        print('Models are the same')
    else:
        print(f'Models have {models_differ} different layers')
