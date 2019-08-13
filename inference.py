# -*- coding: utf-8 -*-

import tensorflow as tf

# 嵌入矩阵的维度
EMBED_DIM = 32

USER_ID_COUNT = 6041
GENDER_COUNT = 2
AGE_COUNT = 7
JOB_COUNT = 21

MOVIE_ID_COUNT = 3953
MOVIE_GENRES_COUNT = 18
MOVIE_TITLE_WORDS_COUNT = 5217

BATCH_SIZE = 256

LSTM_UNIT_NUM = 128


# 用户特征网络核心代码
def user_feature_network(user_id, user_gender, user_age, user_job, dropout_keep_prob, train):
    with tf.variable_scope('user_id_embed'):
        user_id_embed_matrix = tf.get_variable('id_embed_matrix', [USER_ID_COUNT, EMBED_DIM],
                                               initializer=tf.truncated_normal_initializer(stddev=0.1))
        user_embed_layer = tf.nn.embedding_lookup(user_id_embed_matrix, user_id, name='id_lookup')

    with tf.variable_scope('user_gender_embed'):
        gender_embed_matrix = tf.get_variable('gender_embed_matrix', [GENDER_COUNT, EMBED_DIM // 2],
                                              initializer=tf.truncated_normal_initializer(stddev=0.1))
        gender_embed_layer = tf.nn.embedding_lookup(gender_embed_matrix, user_gender, name='gender_lookup')

    with tf.variable_scope('user_age_embed'):
        age_embed_matrix = tf.get_variable('age_embed_matrix', [AGE_COUNT, EMBED_DIM // 2],
                                           initializer=tf.truncated_normal_initializer(stddev=0.1))
        age_embed_layer = tf.nn.embedding_lookup(age_embed_matrix, user_age, name='age_lookup')

    with tf.variable_scope('user_job_embed'):
        job_embed_matrix = tf.get_variable('job_embed_matrix', [JOB_COUNT, EMBED_DIM // 2],
                                           initializer=tf.truncated_normal_initializer(stddev=0.1))
        job_embed_layer = tf.nn.embedding_lookup(job_embed_matrix, user_job, name='job_lookup')

    # user_conv1 = tf.layers.conv1d(user_embed_layer, EMBED_DIM * 2, kernel_size=3, strides=1, padding='same', data_format='channels_first')
    # user_bn1 = tf.layers.batch_normalization(user_conv1, training=train)
    # user_relu1 = tf.nn.relu(user_bn1)
    # user_conv2 = tf.layers.conv1d(user_relu1, EMBED_DIM, kernel_size=3, strides=1, padding='same', data_format='channels_first')
    # user_bn2 = tf.layers.batch_normalization(user_conv2, training=train)
    # user_relu2 = tf.nn.relu(user_bn2)
    user_id_fc_layer = tf.layers.dense(user_embed_layer, EMBED_DIM,
                                       activation=tf.nn.relu,
                                       kernel_regularizer=tf.nn.l2_loss,
                                       kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                       name='user_id_fc')
    user_id_fc_dropout_layer = tf.layers.dropout(user_id_fc_layer, dropout_keep_prob, name='user_id_dropout')

    # gender_conv1 = tf.layers.conv1d(gender_embed_layer, EMBED_DIM * 2, kernel_size=3, strides=1, padding='same',
    #                               data_format='channels_first')
    # gender_bn1 = tf.layers.batch_normalization(gender_conv1, training=train)
    # gender_relu1 = tf.nn.relu(gender_bn1)
    # gender_conv2 = tf.layers.conv1d(gender_relu1, EMBED_DIM, kernel_size=3, strides=1, padding='same',
    #                               data_format='channels_first')
    # gender_bn2 = tf.layers.batch_normalization(gender_conv2, training=train)
    # gender_relu2 = tf.nn.relu(gender_bn2)
    gender_fc_layer = tf.layers.dense(gender_embed_layer, EMBED_DIM,
                                      activation=tf.nn.relu,
                                      kernel_regularizer=tf.nn.l2_loss,
                                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                      name='user_gender_fc')
    gender_fc_dropout_layer = tf.layers.dropout(gender_fc_layer, dropout_keep_prob, name='user_gender_dropout')

    # age_conv1 = tf.layers.conv1d(age_embed_layer, EMBED_DIM * 2, kernel_size=3, strides=1, padding='same',
    #                                 data_format='channels_first')
    # age_bn1 = tf.layers.batch_normalization(age_conv1, training=train)
    # age_relu1 = tf.nn.relu(age_bn1)
    # age_conv2 = tf.layers.conv1d(age_relu1, EMBED_DIM, kernel_size=3, strides=1, padding='same',
    #                                 data_format='channels_first')
    # age_bn2 = tf.layers.batch_normalization(age_conv2, training=train)
    # age_relu2 = tf.nn.relu(age_bn2)
    age_fc_layer = tf.layers.dense(age_embed_layer, EMBED_DIM,
                                   activation=tf.nn.relu,
                                   kernel_regularizer=tf.nn.l2_loss,
                                   kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                   name='user_age_fc')
    age_fc_dropout_layer = tf.layers.dropout(age_fc_layer, dropout_keep_prob, name='user_age_dropout')

    # job_conv1 = tf.layers.conv1d(job_embed_layer, EMBED_DIM * 2, kernel_size=3, strides=1, padding='same',
    #                              data_format='channels_first')
    # job_bn1 = tf.layers.batch_normalization(job_conv1, training=train)
    # job_relu1 = tf.nn.relu(job_bn1)
    # job_conv2 = tf.layers.conv1d(job_relu1, EMBED_DIM, kernel_size=3, strides=1, padding='same',
    #                              data_format='channels_first')
    # job_bn2 = tf.layers.batch_normalization(job_conv2, training=train)
    # job_relu2 = tf.nn.relu(job_bn2)
    job_fc_layer = tf.layers.dense(job_embed_layer, EMBED_DIM,
                                   activation=tf.nn.relu,
                                   kernel_regularizer=tf.nn.l2_loss,
                                   kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                   name='user_job_fc')
    job_fc_dropout_layer = tf.layers.dropout(job_fc_layer, dropout_keep_prob, name='user_job_dropout')

    with tf.name_scope('user_fc'):
        user_combine_feature = tf.concat(
            [user_id_fc_dropout_layer, gender_fc_dropout_layer, age_fc_dropout_layer, job_fc_dropout_layer], 2)
        # user_combine_feature = tf.transpose(user_combine_feature, perm=[0, 2, 1])

        user_combine_conv1 = tf.layers.conv1d(user_combine_feature, EMBED_DIM * 4, kernel_size=3, strides=1, padding='same',
                                     data_format='channels_first')
        user_combine_bn1 = tf.layers.batch_normalization(user_combine_conv1, training=train)
        user_combine_relu1 = tf.nn.relu(user_combine_bn1)
        user_combine_conv2 = tf.layers.conv1d(user_combine_relu1, 1, kernel_size=3, strides=1, padding='same',
                                     data_format='channels_first')
        user_combine_bn2 = tf.layers.batch_normalization(user_combine_conv2, training=train)
        user_combine_relu2 = tf.nn.relu(user_combine_bn2)
        user_combine_fc_layer = tf.layers.dense(user_combine_relu2, 200,
                                                activation=tf.nn.relu,
                                                kernel_regularizer=tf.nn.l2_loss,
                                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                                name='user_fc')
        user_combine_layer_flat = tf.reshape(user_combine_fc_layer, [-1, 200])

    return user_combine_layer_flat


# 电影特征网络核心代码
def movie_feature_embed_network(movie_id, movie_genres):
    with tf.variable_scope('movie_id_embed'):
        movie_id_embed_matrix = tf.get_variable('id_embed_matrix', [MOVIE_ID_COUNT, EMBED_DIM],
                                                initializer=tf.truncated_normal_initializer(stddev=0.1))
        movie_id_embed_layer = tf.nn.embedding_lookup(movie_id_embed_matrix, movie_id, name='id_lookup')

    with tf.name_scope('genres_embed'):
        movie_genres_embed_matrix = tf.Variable(tf.random_uniform([MOVIE_GENRES_COUNT, EMBED_DIM], -1, 1),
                                                name='genres_embed_matrix')

        movie_genres_embed_layer = tf.matmul(movie_genres, movie_genres_embed_matrix)
        movie_genres_embed_layer = tf.expand_dims(movie_genres_embed_layer, 1)

    return movie_id_embed_layer, movie_genres_embed_layer


def movie_title_lstm_layer(movie_titles, movie_title_length, dropout_keep_prob):
    with tf.variable_scope('movie_title_embed'):
        movie_title_embed_matrix = tf.get_variable('title_embed_matrix', [MOVIE_TITLE_WORDS_COUNT, EMBED_DIM],
                                                   initializer=tf.truncated_normal_initializer(stddev=0.1))
        movie_title_embed_layer = tf.nn.embedding_lookup(movie_title_embed_matrix, movie_titles,
                                                         name='title_lookup')

    lstm_cell = tf.contrib.rnn.BasicLSTMCell(LSTM_UNIT_NUM, forget_bias=0.0)

    with tf.name_scope("movie_title_dropout"):
        lstm_cell_dropout = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=dropout_keep_prob)

        # 根据输入动态决定对应的batch_size大小
        batch_size_ = tf.shape(movie_titles)[0]
        init_state = lstm_cell_dropout.zero_state(batch_size_, dtype=tf.float32)

    # 步长根据标题长度动态变化，dynamic_rnn会将填充长度输出置为0
    lstm_output, final_state = tf.nn.dynamic_rnn(lstm_cell_dropout,
                                                 movie_title_embed_layer,
                                                 sequence_length=movie_title_length,
                                                 initial_state=init_state,
                                                 scope='movie_title_rnn')
    # 根据标题长度计算平均值，除数是标题的真实长度
    with tf.name_scope('movie_title_avg_pool'):
        lstm_output = tf.reduce_sum(lstm_output, 1) / movie_title_length[:, None]

    return lstm_output


def movie_feature_network(movie_id, movie_genres, movie_titles, movie_title_length, dropout_keep_prob, train):
    movie_id_embed_layer, movie_genres_embed_layer = movie_feature_embed_network(movie_id, movie_genres)

    # movie_id_conv1 = tf.layers.conv1d(movie_id_embed_layer, EMBED_DIM * 2, kernel_size=3, strides=1, padding='same',
    #                               data_format='channels_first')
    # movie_id_bn1 = tf.layers.batch_normalization(movie_id_conv1, training=train)
    # movie_id_relu1 = tf.nn.relu(movie_id_bn1)
    # movie_id_conv2 = tf.layers.conv1d(movie_id_relu1, EMBED_DIM, kernel_size=3, strides=1, padding='same',
    #                               data_format='channels_first')
    # movie_id_bn2 = tf.layers.batch_normalization(movie_id_conv2, training=train)
    # movie_id_relu2 = tf.nn.relu(movie_id_bn2)
    movie_id_fc_layer = tf.layers.dense(movie_id_embed_layer, EMBED_DIM,
                                        activation=tf.nn.relu,
                                        kernel_regularizer=tf.nn.l2_loss,
                                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                        name='movie_id_fc')
    movie_id_dropout_layer = tf.layers.dropout(movie_id_fc_layer, dropout_keep_prob, name='movie_id_dropout')

    # movie_genres_conv1 = tf.layers.conv1d(movie_genres_embed_layer, EMBED_DIM * 2, kernel_size=3, strides=1, padding='same',
    #                                   data_format='channels_first')
    # movie_genres_bn1 = tf.layers.batch_normalization(movie_genres_conv1, training=train)
    # movie_genres_relu1 = tf.nn.relu(movie_genres_bn1)
    # movie_genres_conv2 = tf.layers.conv1d(movie_genres_relu1, EMBED_DIM, kernel_size=3, strides=1, padding='same',
    #                                   data_format='channels_first')
    # movie_genres_bn2 = tf.layers.batch_normalization(movie_genres_conv2, training=train)
    # movie_genres_relu2 = tf.nn.relu(movie_genres_bn2)
    movie_genres_fc_layer = tf.layers.dense(movie_genres_embed_layer, EMBED_DIM,
                                            activation=tf.nn.relu,
                                            kernel_regularizer=tf.nn.l2_loss,
                                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                            name='movie_genres_fc')
    movie_genres_dropout_layer = tf.layers.dropout(movie_genres_fc_layer, dropout_keep_prob,
                                                   name='movie_genres_dropout')

    # 获取电影名的特征向量
    movie_title_output_layer = movie_title_lstm_layer(movie_titles, movie_title_length, dropout_keep_prob)
    movie_title_output_layer = tf.expand_dims(movie_title_output_layer, 1)

    with tf.name_scope('movie_fc_layer'):
        # movie_id_dropout_layer = tf.reduce_sum(movie_id_dropout_layer, 1)
        movie_combine_feature = tf.concat(
            [movie_id_dropout_layer, movie_genres_dropout_layer, movie_title_output_layer], 2)

        movie_combine_conv1 = tf.layers.conv1d(movie_combine_feature, EMBED_DIM * 4, kernel_size=3, strides=1,
                                              padding='same',
                                              data_format='channels_first')
        movie_combine_bn1 = tf.layers.batch_normalization(movie_combine_conv1, training=train)
        movie_combine_relu1 = tf.nn.relu(movie_combine_bn1)
        movie_combine_conv2 = tf.layers.conv1d(movie_combine_relu1, 1, kernel_size=3, strides=1, padding='same',
                                              data_format='channels_first')
        movie_combine_bn2 = tf.layers.batch_normalization(movie_combine_conv2, training=train)
        movie_combine_relu2 = tf.nn.relu(movie_combine_bn2)
        movie_combine_layer = tf.layers.dense(movie_combine_relu2, 200,
                                              activation=tf.nn.relu,
                                              kernel_regularizer=tf.nn.l2_loss,
                                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                              name='movie_fc_layer')
        movie_combine_layer = tf.reshape(movie_combine_layer, [-1, 200])
    return movie_combine_layer


# 损失层核心代码
def full_network(uid, user_gender, user_age, user_job, movie_id, movie_genres, movie_titles, movie_title_length,
                 dropout_keep_prob, train):
    # 得到用户特征
    user_combine_layer_flat = user_feature_network(uid, user_gender, user_age, user_job, dropout_keep_prob, train)
    # 获取电影特征
    movie_combine_layer = movie_feature_network(movie_id, movie_genres, movie_titles, movie_title_length,
                                                dropout_keep_prob, train)
    # 将用户特征和电影特征作为输入，经过全连接，输出一个值
    with tf.name_scope('user_movie_fc'):
        input_layer = tf.concat([user_combine_layer_flat, movie_combine_layer], 1)  # (?, 200)
        predicted = tf.layers.dense(input_layer, 1,
                                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                    kernel_regularizer=tf.nn.l2_loss,
                                    name='user_movie_fc')

    return user_combine_layer_flat, movie_combine_layer, predicted


def get_inputX(uid, user_gender, user_age, user_job, movie_id, movie_genres, movie_titles, movie_title_length,
               dropout_keep_prob):
    # 得到用户特征
    user_combine_layer_flat = user_feature_network(uid, user_gender, user_age, user_job, dropout_keep_prob)
    # 获取电影特征
    movie_combine_layer = movie_feature_network(movie_id, movie_genres, movie_titles, movie_title_length,
                                                dropout_keep_prob)
    # 将用户特征和电影特征作为输入，经过全连接，输出一个值
    with tf.name_scope('user_movie_fc'):
        input_X = tf.concat([user_combine_layer_flat, movie_combine_layer], 1)
    return input_X


def trainable_variable_summaries():
    for variable in tf.trainable_variables():
        name = variable.name.split(':')[0]
        tf.summary.histogram(name, variable)
        mean = tf.reduce_mean(variable)
        tf.summary.scalar('mean/' + name, mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(variable - mean)))
        tf.summary.scalar('stddev/' + name, stddev)
