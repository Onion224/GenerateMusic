# -*- codeing = utf-8 -*-
# @Time :2021/5/26 9:34
# @Author :Onion
# @File :melodyGenerator.py
# @Software :PyCharm
import tensorflow.keras as keras
import json
import numpy as np
import music21 as m21
from music21 import *
from preprocess import SEQUENCE_LENGTH, MAPPING_PATH


class MelodyGenerator:
    def __init__(self, model_path="model.h5"):
        # 导入模型
        self.model_path = model_path
        self.model = keras.models.load_model(model_path)

        with open(MAPPING_PATH, "r") as fp:
            # 加载映射文件
            self.mappings = json.load(fp)

        self._start_symbols = ["/"] * SEQUENCE_LENGTH

    # seed是一段初始化的小旋律例如"64 _ 63 _ _",num_steps是我们想要神经网络输出的长度

    def generate_melody(self, seed, num_steps, max_sequence_length, temperature):

        # 创造一个种子
        seed = seed.split()
        melody = seed
        seed = self._start_symbols + seed

        # 将种子旋律映射成Integer
        seed = [self.mappings[symbol] for symbol in seed]

        for _ in range(num_steps):

            # 限制种子在max长度之内
            seed = seed[-max_sequence_length:]

            # 将seed编码为one_hot格式
            onehot_seed = keras.utils.to_categorical(seed, num_classes=len(self.mappings))
            onehot_seed = onehot_seed[np.newaxis, ...]

            # 进行预测
            probabilities = self.model.predict(onehot_seed)[0]

            output_int = self._sample_with_temperature(probabilities, temperature)

            # update seed
            seed.append(output_int)

            # map int to our encoding
            output_symbol = [k for k, v in self.mappings.items() if v == output_int][0]

            # check whether we're at the end of a melody
            if output_symbol == "/":
                break

            # update melody
            melody.append(output_symbol)

        return melody

    def _sample_with_temperature(self, probabilites, temperature):
        """Samples an index from a probability array reapplying softmax using temperature
        :param predictions (nd.array): Array containing probabilities for each of the possible outputs.
        :param temperature (float): Float in interval [0, 1]. Numbers closer to 0 make the model more deterministic.
            A number closer to 1 makes the generation more unpredictable.
        :return index (int): Selected output symbol
        """
        predictions = np.log(probabilites) / temperature
        probabilites = np.exp(predictions) / np.sum(np.exp(predictions))

        choices = range(len(probabilites))  # [0, 1, 2, 3]
        index = np.random.choice(choices, p=probabilites)

        return index

# 将旋律保存为midi文件
    def save_melody(self, melody, step_duration=0.25, format="midi", file_name="mel.mid"):
        """Converts a melody into a MIDI file
        :param melody (list of str):
        :param min_duration (float): Duration of each time step in quarter length
        :param file_name (str): Name of midi file
        :return:
        """

        # create a music21 stream
        stream = m21.stream.Stream()

        start_symbol = None
        step_counter = 1

        # parse all the symbols in the melody and create note/rest objects
        for i, symbol in enumerate(melody):

            # handle case in which we have a note/rest
            if symbol != "_" or i + 1 == len(melody):

                # ensure we're dealing with note/rest beyond the first one
                if start_symbol is not None:

                    quarter_length_duration = step_duration * step_counter # 0.25 * 4 = 1

                    # handle rest
                    if start_symbol == "r":
                        m21_event = m21.note.Rest(quarterLength=quarter_length_duration)

                    # handle note
                    else:
                        m21_event = m21.note.Note(int(start_symbol), quarterLength=quarter_length_duration)

                    stream.append(m21_event)

                    # reset the step counter
                    step_counter = 1

                start_symbol = symbol

            # handle case in which we have a prolongation sign "_"
            else:
                step_counter += 1

        # write the m21 stream to a midi file
        stream.write(format, file_name)

if __name__ == "__main__":
    melodyGenerator = MelodyGenerator()
    seed = "55 _ _ _ 60 _ _ _ 55 _ _ _ 55 _"
    melody = melodyGenerator.generate_melody(seed, 500, SEQUENCE_LENGTH, 0.7)
    print(melody)
    melodyGenerator.save_melody(melody)




