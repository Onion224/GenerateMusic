#-*- codeing = utf-8 -*-
#@Time :2021/5/23 12:00
#@Author :Onion
#@File :preprocess.py.py
#@Software :PyCharm

import os
import json
# keras用来构建One-hot格式的input和构建神经网络
import tensorflow.keras as keras
import numpy as np
#使用music21将krn文件转换
import music21 as m21
from music21 import *

environment.Environment()['musicxmlPath'] = r'D:\Musescore\bin\MuseScore3.exe'
environment.Environment()['musescoreDirectPNGPath'] = r'D:\Musescore\bin\MuseScore3.exe'


KERN_DATASET_PATH = "deutschl/erk"
SAVE_PATH = "dataset"
MAPPING_PATH = "mapping.json"
SINGLE_FILE_DATASET = "file_dataset"
SEQUENCE_LENGTH = 64
# 表示可以接受的音符范围,以4分音符为基准
ACCEPTABLE_DURATIONS = [
    0.25, # 16分音符
    0.5, # 8分音符
    0.75,
    1.0, # 4分音符
    1.5,
    2, # 二分音符
    3,
    4 # 全音符
]
# 导入歌曲
def load_song_in_krn(dataset_path):

    # 用一个list来保存所有加载到的歌曲
    songs = []

    # 遍历数据集中所有的文件,并用music21加载它们
    for path,subdirs,files in os.walk(dataset_path):
        for file in files:
            if file[-3:] == "krn":
                # 这里的song是m21的一个乐谱实例
                song = m21.converter.parse(os.path.join(path,file))
                songs.append(song)
    # 将加载到的歌曲返回
    return songs

# 过滤不常用的音符(音符代表的是音的持续时间)
def has_acceptable_durations(song,acceptable_durations):
    for note in song.flat.notesAndRests:
        if note.duration.quarterLength not in acceptable_durations:
            return False
    return True

# 将歌曲转换成C大调或A小调
def transpose(song):
    # 获取歌曲的key(调)
    parts = song.getElementsByClass(m21.stream.Part)
    measure_part0 = parts[0].getElementsByClass(m21.stream.Measure)
    key = measure_part0[0][4]
    # 使用music21评估歌曲的key
    if not isinstance(key,m21.key.Key):
        key = song.analyze("key")
    # 实现调的转换    例如: Bmaj-->Cmaj(B大调转换为C大调)
    # 如果是大调,转换为C大调,tonic表示主音
    if key.mode == "major":
        interval = m21.interval.Interval(key.tonic,m21.pitch.Pitch("C"))
    # 如果是小调,转换为A小调
    elif key.mode == "minor":
        interval = m21.interval.Interval(key.tonic,m21.pitch.Pitch("A"))

    # transpose song by calculated interval
    transpose_song = song.transpose(interval)

    # 检验转换后的key
    # parts = transpose_song.getElementsByClass(m21.stream.Part)
    # measure_part1 = parts[0].getElementsByClass(m21.stream.Measure)
    # key2 = measure_part1[0][4]
    # print(key2)   # 打印的结果是a minor,的确将e小调转换成了a小调
    return transpose_song

# 将音乐数据转换为时间序列格式,并返回字符串的格式
# 四分音符的表示为1,1/0.25 = 4,所以time_step=0.25
def encode_song(song,time_step = 0.25):

    encoded_song = []

    for event in song.flat.notesAndRests:

        # handle notes
        if isinstance(event, m21.note.Note):
            symbol = event.pitch.midi # 60
        # handle rests
        elif isinstance(event, m21.note.Rest):
            symbol = "r"

        # convert the note/rest into time series notation
        # ["r", "_", "60", "_", "_", "_", "72" "_"],这里的steps = 8
        steps = int(event.duration.quarterLength / time_step)
        for step in range(steps):

            # if it's the first time we see a note/rest, let's encode it. Otherwise, it means we're carrying the same
            # symbol in a new time step
            if step == 0:
                encoded_song.append(symbol)
            else:
                encoded_song.append("_")

    # cast encoded song to str,使用map函数将encoded_song的所有元素转换为string之后用join连接成一个长的字符串
    encoded_song = " ".join(map(str, encoded_song))

    # 将转换之后的字符串返回
    return encoded_song

def load(file_path):
    with open(file_path, "r") as fp:
        song = fp.read()
    return song

# 为了方便将数据传入LSTM网络中,需要将所有的歌曲数据合成为一个单一的文件,里面包含了数据集中的所有歌曲
# 并且,使用特定的符号'/'来分隔不同的歌曲,将合成的文件保存在file_dataset_path
def create_single_file_dataset(dataset_path,file_dataset_path,sequence_length):
    # 用来分隔不同歌曲的字符串序列
    new_song_delimiter = "/ "*sequence_length
    songs = ""

    # 加载已经转换为时间序列格式的歌曲,并且为每首歌之前添加(sequence_length)个分隔符"/"
    for path, _, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(path, file)
            song = load(file_path)
            songs = songs + song + " " + new_song_delimiter

    # 将字符串最后的空字符去除掉
    songs = songs[:-1]

    # 保存包含所有歌曲的字符串到文件中
    with open(file_dataset_path, "w") as fp:
        fp.write(songs)

    return songs


# 建立文件中所有符号的映射,使用json来进行存储
def create_mapping(songs, mapping_path):
    mappings = {}

    # 使用split函数将songs中的每一个元素分割,得到2576项元素
    songs = songs.split()
    # 使用set进行去重,将2576个元素去重后将去重后的元素用一个list存取
    vocabulary = list(set(songs))

    # create mappings
    for i, symbol in enumerate(vocabulary):
        # 为list中每一个已经去重后的元素建立映射,mapping的key为元素的值,value为遍历变量i
        mappings[symbol] = i

    # save voabulary to a json file
    with open(mapping_path, "w") as fp:
        # indent = 4让写入的json能够以竖着的形式展示
        # 将mappings写入fp所指的mapping.json,以json格式存储
        json.dump(mappings, fp, indent=4)


#  将字符串格式的歌曲集合songs利用mapping转换为能够输入进神经网络的int格式
def convert_songs_to_int(songs):

    int_songs = []

    # 加载映射关系文件
    with open(MAPPING_PATH,"r") as fp:
        mappings = json.load(fp)
    # 将string格式的songs中的元素用list来存储,以方便遍历
    songs = songs.split()

    # 遍历分割存储后的songs(list)映射为int
    for symbol in songs:
        int_songs.append(mappings[symbol])

    #   这样就将songs(list)->int_songs(list)
    return int_songs

#   生成输入LSTM的训练序列,每一次用sequence_length长度的数据作为一次input,得到一个targets
def generate_training_sequences(sequence_length):

    songs = load(SINGLE_FILE_DATASET)

    int_songs = convert_songs_to_int(songs)

    # 用来保存One-host格式的input
    inputs = []
    targets = []
    # 生成训练序列,用个数为num_sequence,长度为sequence_length的inputs,得到个数为num_sequence,长度为一个音的输出targets
    num_sequence = len(int_songs) - sequence_length
    for i in range(num_sequence):
        inputs.append(int_songs[i:i+sequence_length])
        targets.append(int_songs[i+sequence_length])

    # 将输入数据inputs转换为One-hot
    vocabulary_size = len(set(int_songs))
    inputs = keras.utils.to_categorical(inputs, num_classes=vocabulary_size)

    # 输出数据用数组保存,list->np.array
    targets = np.array(targets)

    return inputs,targets
# 数据预处理
def preprocess(dataset_path):
    # 步骤
    # 一：导入歌曲
    print("loding songs...")
    songs = load_song_in_krn(dataset_path)
    print(f"Loaded{len(songs)} songs.")
    # 二：过滤不可用歌曲(歌曲中包含的音符不在所限定的范围之内)
    # 通过enumerate函数可以拿到index
    for i,song in enumerate(songs):
        # 过滤歌曲中包含的音符不在所限定的范围之内
        if not has_acceptable_durations(song, ACCEPTABLE_DURATIONS):
            continue
        # 三：将歌曲转换为C大调或A小调(如果考虑所有的24种情况的话成本太高,因此全部都转换成C大调或者A小调以方便学习)
        song = transpose(song)
        # 四：将歌曲编码为时间序列格式
        encoded_song = encode_song(song)
        # 五：将歌曲保存在文本文件(text)中,并用下标值i来为文件命名
        save_path = os.path.join(SAVE_PATH,str(i))
        with open(save_path,"w") as fp:
            fp.write(encoded_song)


def main():
    preprocess(KERN_DATASET_PATH)
    songs = create_single_file_dataset(SAVE_PATH,SINGLE_FILE_DATASET,SEQUENCE_LENGTH)
    create_mapping(songs,MAPPING_PATH)

if __name__ == "__main__":
    main()