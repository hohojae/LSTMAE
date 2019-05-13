import tensorflow as tf
import numpy as np
import os
import RNN_AE_model_decoder_feedback as rnn_AE
import utils
import csv
import time

with open("nowtime.txt", 'r') as f: # nowtime.txt가 파일의 경로
    savepath = f.read()
sv_datetime = savepath

util = utils.Util()
MIN_SONG_LENGTH = 35
ms = time.strftime('_%H%M%S', time.localtime(time.time()))
filename = sv_datetime
filename = filename + ms

def test(trained_data, len_data, mode): # trained_data = song_pitches or song_durations, len_data = MIN_SONG_LENGTH, # mode = pitch or duration
    # Test the RNN model
    char2idx = util.getchar2idx(mode=mode)

    enc_model = rnn_AE.LSTMAutoEnc(sequence_length=len_data,
                                   batch_size=len(trained_data),
                                   mode='test')
    dec_model = rnn_AE.LSTMAutoEnc(sequence_length=160,
                                   batch_size=1,
                                   mode='test')

    enc_out_state = enc_model.encoder(scopename=mode)
    dec_model.decoder(scopename=mode)

    # encoder input
    state_sample_data = util.data2idx(trained_data, char2idx)

    enc_saver = tf.train.Saver(var_list=enc_model.enc_vars)
    dec_saver = tf.train.Saver(var_list=dec_model.dec_vars)
    with tf.Session() as sess:
        enc_saver.restore(sess, "./save/" + sv_datetime + "/enc_{}_model.ckpt".format(mode))
        dec_saver.restore(sess, "./save/" + sv_datetime + "/dec_{}_model.ckpt".format(mode))

        #enc_out_state = sess.run(enc_out_state, feed_dict={enc_model.Enc_input: state_sample_data})

        # avarage encoder state output
        #dec_in_state = np.mean(enc_out_state, 0).reshape([dec_model.batch_size, dec_model.enc_cell.state_size])

        random_input_flag = 1
        if random_input_flag == 1:
            # random vector input in decoder initial state
            #np.random.seed(100)
            dec_in_state = np.random.randn(1, dec_model.enc_cell.state_size)

        prediction = sess.run(dec_model.prediction, feed_dict={dec_model.Dec_state: dec_in_state})

        result = util.idx2char(prediction, mode)
        print("result : ", result)

        # print : result - trained_data
        print_error(result, trained_data, mode)

        return result

def print_error(result, trained_data, mode):
    # print : result - trained_data
    trained_data = trained_data[0]
    result = result[:len(trained_data)]
    print("trained_data : ", trained_data)
    if mode == 'pitch':
        result_mean = []
        result_std = []
        cos_similarity = []
        pearson_correlation = []
        result = result[1:]
        trained_data = trained_data[1:]
        for i in range(0, len(result)): # result에서 'Rest'값을 trained_data의 mean값으로 대치
            if result[i] == 'Rest':
                result[i] = np.mean(trained_data)
        error = [abs(int(x) - int(y)) for x, y in zip(result, trained_data)]
        cos_similarity.append(cos_sim(result, trained_data))
        result_mean.append(np.mean(result))
        result_std.append(np.std(result))
        pearson_correlation.append(pearson_cor(result, trained_data))
        print("error : ", error)
        print("total error : ", sum(error))
        print("cosine similarity : %0.3f" % cos_sim(result, trained_data))
        print("mean : %0.3f" % np.mean(result))
        print("std : %0.3f" % np.std(result))
        print("pearson correlation : %0.3f" % pearson_cor(result, trained_data))
        # mean_sub과 std_sub을 csv에 저장
        if (os.path.exists("./mean_std/mean_std.csv")) == False:
            with open('./mean_std/mean_std.csv', 'w', newline='') as f:
                wr = csv.writer(f)
                wr.writerow(result_mean)
                wr.writerow(result_std)
        else:
            with open('./mean_std/mean_std.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(result_mean)
                writer.writerow(result_std)
        # 생성된 곡의 cosine similarity와 pearson correlation 저장
        if (os.path.exists("./cos_sim/cos_sim_pearson.csv")) == False:
            with open('./cos_sim/cos_sim_pearson.csv', 'w', newline='') as f:
                wr = csv.writer(f)
                wr.writerow(cos_similarity)
                wr.writerow(pearson_correlation)
        else:
            with open('./cos_sim/cos_sim_pearson.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(cos_similarity)
                writer.writerow(pearson_correlation)
    else:
        result_mean = []
        result_std = []
        cos_similarity = []
        pearson_correlation = []
        result = result[1:]
        trained_data = trained_data[1:]
        for i in range(0, len(result)):
            if result[i] == 'Rest':
                result[i] = np.mean(trained_data)
        error = [abs(x - y) for x, y in zip(result, trained_data)]
        cos_similarity.append(cos_sim(result, trained_data))
        result_mean.append(np.mean(result))
        result_std.append(np.std(result))
        pearson_correlation.append(pearson_cor(result, trained_data))
        print("error : ", error)
        print("total error : ", sum(error))
        print("cosine similarity : %0.3f" % cos_sim(result, trained_data))
        print("mean : %0.3f" % np.mean(result))
        print("std : %0.3f" % np.std(result))
        print("pearson correlation : %0.3f" % pearson_cor(result, trained_data))

# 평가지표 1 : 코사인 유사도 (1에 가까우면 출력곡과 훈련데이터의 유사도가 높다는 것을 의미)
def cos_sim(result, trained_data):
    result = np.array(result)
    trained_data = np.array(trained_data)
    result = result[:trained_data.shape[0]]
    result.shape, trained_data.shape
    cosine_similarity = np.dot(result, trained_data)/(np.linalg.norm(result)*np.linalg.norm(trained_data))
    return cosine_similarity
# 평가지표 2 : 피어슨 상관계수
def pearson_cor(result, trained_data):
    result = np.array(result)
    trained_data = np.array(trained_data)
    result = result[:trained_data.shape[0]]
    result.shape, trained_data.shape
    a = (len(result)-1) * np.std(result) * np.std(trained_data)
    b = 0
    for i in range(len(result)):
        b += (result[i]-np.mean(result))*(trained_data[i]-np.mean(trained_data))
    pearson_correlation = b/a
    return pearson_correlation

'''
def songs_load():
    filenames = ['988-v07.mid', '988-v08.mid', '988-v12.mid']
    trained_songs = []
    for f in filenames:
        trained_songs.append(util.get_one_song(f))
    return trained_songs
'''

def main(_):

    songs = util.get_all_song()

    print("Load {} Songs...".format(len(songs)))
    songs_name = ""
    songs_len = []
    songs_pitches = []
    songs_durations = []
    for song in songs:
        '''
        print("name : ", song['name'])
        print("length : ", song['length'])
        print("pitches : ", song['pitches'])
        print("durations : ", song['durations'])
        print("")
        '''
        songs_name += "_" + song['name']
        songs_len.append(song['length'])
        songs_pitches.append(song['pitches'])
        songs_durations.append(song['durations'])

    # 여러 곡의 길이를 제일 짧은 곡에 맞춘다.
    for i in range(len(songs_pitches)):
        if len(songs_pitches[i]) > MIN_SONG_LENGTH: # min(songs_len)
            songs_pitches[i] = songs_pitches[i][:MIN_SONG_LENGTH]
            songs_durations[i] = songs_durations[i][:MIN_SONG_LENGTH]

    # output song
    pitches = test(songs_pitches, MIN_SONG_LENGTH, mode='pitch')
    durations = test(songs_durations, MIN_SONG_LENGTH, mode='duration')

    # make midi file
    util.song2midi(pitches, durations, './generate', filename)


if __name__ == '__main__':
    tf.app.run()


