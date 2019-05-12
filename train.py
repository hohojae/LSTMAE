# basic RNN using one-hot encdoing time series song data
import os
import random
import tensorflow as tf
import numpy as np
from datetime import datetime
import RNN_AE_model_decoder_feedback as rnn_AE
from utils import Util
import csv
import matplotlib.pyplot as plt

util = Util()
now = datetime.now()
NOWTIME = now.strftime("%Y%m%d-%H%M") # 파일 생성용
nowTIME = np.int(now.strftime("%d%H%M%S")) # 그래프 생성용
lossfile = "./loss/loss_%d.csv" %nowTIME
step = 2000 # training step은 2000으로 잡고 10마다 loss저장
with open("nowtime.txt", 'w') as f:
    f.write(NOWTIME)

def train(trained_data, model, mode):
    '''
    train the model using data
    :param trained_song: list, Song data
        model: class, NN Model
        mode: string, "pitches" or "durations"
    :return:
    '''
    print('Start Train : {}'.format(mode)) # mode = pitch or duration

    model.build(scopename=mode) # mode = pitch or duration

    # make char2idx
    char2idx = util.getchar2idx(mode=mode)

    enc_saver = tf.train.Saver(var_list=model.enc_vars)
    dec_saver = tf.train.Saver(var_list=model.dec_vars)
    loss_plot_pitch = []
    step_plot_pitch = []

    with tf.Session() as sess:
        writer = tf.summary.FileWriter("./graphs/" + NOWTIME, graph=sess.graph)
        sess.run(tf.global_variables_initializer())

        for i in range(step):
            train_samples = [trained_data[i] for i in sorted(random.sample(range(len(trained_data)), model.batch_size))] # 배치사이즈(지금은 5)만큼 랜덤으로 훈련할곡 뽑음
            x_data = util.data2idx(train_samples, char2idx)
            x_data = np.reshape(x_data, [model.batch_size, x_data.shape[1]]) #

            loss_val, _ = sess.run([model.loss, model.train], feed_dict={model.Enc_input: x_data}) # train
            result = sess.run(model.prediction, feed_dict={model.Enc_input: x_data})
            summary = sess.run(model.summary_op, feed_dict={model.Enc_input: x_data})
            writer.add_summary(summary, i)

            #result_str = [model.idx2char[c] for c in np.squeeze(result)]
            print("{:4d}  loss: {:.5f}".format(i, loss_val))
            # pitch의 loss를 csv파일에 저장
            if mode == "pitch":
                if (i % 10) == 0:
                    loss_plot_pitch.append(loss_val)
                    step_plot_pitch.append(i)
        with open(lossfile, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(loss_plot_pitch)
            writer.writerow(step_plot_pitch)

        dir_path = "./save/"
        dir_name = NOWTIME
        enc_saver.save(sess, "./save/" + NOWTIME + "/enc_{}_model.ckpt".format(mode))
        dec_saver.save(sess, "./save/" + NOWTIME + "/dec_{}_model.ckpt".format(mode))


def main(_):
    ### Song setting ###
    # load one midi file
    filename = 'test.mid'
    trained_song = util.get_one_song(filename)
    print("One song load : {}".format(filename))
    '''
    print("name : ", trained_song['name'])
    print("length : ", trained_song['length'])
    print("pitches : ", trained_song['pitches'])
    print("durations : ", trained_song['durations'])
    '''
    # load all midi file
    all_songs = util.get_all_song()

    songs = all_songs
    print("Load {} Songs...".format(len(songs))) # 송의 개수 출력
    songs_len = []
    songs_pitches = []
    songs_durations = []
    for song in songs: # 송들의 정보 출력
        print("name : ", song['name'])
        print("length : ", song['length'])
        print("pitches : ", song['pitches'])
        print("durations : ", song['durations'])
        print("")

        if song['length'] < 10 :
            util.delete_empty_song(song['name'])
            continue

        songs_len.append(song['length'])
        songs_pitches.append(song['pitches']) # 노래들의 피치 저장
        songs_durations.append(song['durations'])

    # 여러 곡의 길이를 제일 짧은 곡에 맞춘다.
    for i in range(len(songs_pitches)):
        if len(songs_pitches[i]) > min(songs_len):
            songs_pitches[i] = songs_pitches[i][:min(songs_len)]
            songs_durations[i] = songs_durations[i][:min(songs_len)]

    ### Train setting ###
    num_songs = len(songs) # num_songs = 노래 개수
    num_melody = min(songs_len) # num_melody = 가장 짧은 노래의 길이

    print("num_song: ", num_songs)
    print("num_melody: ", num_melody)

    #print("melody max: ", np.array().max())
    #print("melody min: ", np.array(songs_pitches).min())

    # pitch net
    pitch_net = rnn_AE.LSTMAutoEnc(sequence_length=num_melody, # 시퀀스렝스=노래길이
                                   batch_size=5,
                                   mode='train')
    # duration net
    duration_net = rnn_AE.LSTMAutoEnc(sequence_length=num_melody,
                                      batch_size=5,
                                      mode='train')
    # train NN
    if num_songs == 1:
        train([trained_song['pitches'][:num_melody]], pitch_net, mode='pitch')
        train([trained_song['durations'][:num_melody]], duration_net, mode='duration')
    else:
        train(songs_durations, duration_net, mode='duration') # duration 먼저해야 pitch의 loss.csv 얻을 수 있음.
        train(songs_pitches, pitch_net, mode='pitch')
    # loss를 그래프로 그려서 저장
    with open(lossfile, 'r') as f:
        x_list = []
        y_list = []
        rdr = csv.reader(f)
        csv_list = list(rdr)
        x = map(int, csv_list[1])
        y = map(float, csv_list[0])
        for t in x:
            x_list.append(t)
        for i in y:
            y_list.append(i)
        plt.plot(x_list, y_list, color="black")
        plt.xlabel('Training step')
        plt.ylabel('Loss')
        plt.title('LSTM-Autoencoder Loss')
        plt.grid()
        plt.savefig("./loss/loss_%d_LAE.png" % nowTIME)
        # plt.show()

if __name__ == '__main__':
    tf.app.run()
