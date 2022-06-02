import numpy as np
import config
import model
import read_data


def data_reader(train=False, val=False, test=False):
    data_dic = dict()
    if train:
        training_input, training_type_one_hot, training_type = read_data.training_read(config.training_data_path)
        data_dic.update(zip(["training_input", "training_type_one_hot", "training_type"],
                            [training_input, training_type_one_hot, training_type]))
    if val:
        validating_input, validating_type_one_hot, validating_type = read_data.validating_read(config.validate_data_path)
        data_dic.update(zip(["validating_input", "validating_type_one_hot", "validating_type"],
                            [validating_input, validating_type_one_hot, validating_type]))
    if test:
        testing_input, testing_type_one_hot, testing_type = read_data.testing_read(config.testing_data_path)
        data_dic.update(zip(["testing_input", "testing_type_one_hot", "testing_type"],
                            [testing_input, testing_type_one_hot, testing_type]))
    return data_dic


if __name__ == '__main__':
    dic_data = data_reader(train=True, val=False, test=False)

    Transformer_model = model.Transformer_model()
    Transformer_model.train(encoder_inputs=dic_data["training_input"],
                            decoder_inputs=dic_data["training_input"],
                            y_train=dic_data["training_type"])
