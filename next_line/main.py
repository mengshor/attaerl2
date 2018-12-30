# -*- coding: utf-8 -*-
import GetResult as GR
import numpy as np
from keras.layers import Input, Dense, Lambda, multiply, Layer, add, concatenate, Flatten
from keras.models import Model
from keras import backend as K
from keras import objectives
from util import get_weighted
from keras.callbacks import EarlyStopping


train_doc = 'x_train/doc.npy'
train_pho0 = 'x_train/pho_0.npy'
train_pho1 = 'x_train/pho_1.npy'
train_y = 'x_train/y.npy'
#
test_num = 10000
test_doc = 'x_test/doc.npy'
test_pho0 = 'x_test/pho_0.npy'
test_pho1 = 'x_test/pho_1.npy'


# load training dataset
x_train_doc = np.load(train_doc)  # doc2vec feature file
x_train0 = np.load(train_pho0)
x_train1 = np.load(train_pho1)
y_train = np.load(train_y)

# load testing dataset
x_test_doc = np.load(test_doc)  # doc2vec feature file
x_test0 = np.load(test_pho0)  # feature file
x_test1 = np.load(test_pho1)

pho_dim = 125
doc_dim = 125
con_dim = 250

activ = 'tanh'
optim = 'sgd'
binary = False


def tensor_slice(x, start, end):
    return x[:, start:end]


def generate_batch_data_random(x, y, batch_size):
    ylen = len(y)
    loopcount = (ylen + batch_size - 1) // batch_size
    # idx = shuffle(list(range(loopcount + 1)))
    while (True):
        for i in range(loopcount):
            yield [x[0][i * batch_size:min((i + 1) * batch_size, ylen)], x[1][
                                                                         i * batch_size:min((i + 1) * batch_size,
                                                                                            ylen)],
                   x[2][i * batch_size:min((i + 1) * batch_size, ylen)], y[
                                                                         i * batch_size:min((i + 1) * batch_size,
                                                                                            ylen)]], None


class CustomVariationalLayer(Layer):
    def __init__(self, paras, **kwargs):
        self.is_placeholder = True
        self.hyperparas = paras
        super(CustomVariationalLayer, self).__init__(**kwargs)

    def loss(self, losses):
        l = sum([p * losses[i] for i, p in enumerate(self.hyperparas)])
        return K.mean(l)

    def call(self, inputs):
        l = self.loss(inputs)
        self.add_loss(l, inputs=inputs)
        # We won't actually use the output.
        return inputs[0]


def rhyme2vec(alpha=0, beta=0, activation=activ, use_bias=True, latent_dim=100, epochs=5, batch_size=10000):
    # recall@k
    rec_k_list = [1, 5, 30, 150]
    # the input dimension
    input_shape = pho_dim

    print('====load dataset done====' + '\n')

    x_0 = Input(shape=(input_shape,))
    x_1 = Input(shape=(input_shape,))
    x, att = get_weighted([x_0, x_1], pho_dim)
    x = Dense(units=latent_dim, activation=activation, use_bias=use_bias)(x)
    sig = Dense(1, activation='sigmoid', use_bias=use_bias)(x)

    rhyme = Model(outputs=sig, inputs=[x_0, x_1])
    print('=======Model Information=======' + '\n')
    rhyme.summary()
    rhyme.compile(optimizer=optim, loss='binary_crossentropy')
    rhyme.fit([x_train0, x_train1], y_train,
              shuffle=False,
              epochs=epochs,
              batch_size=batch_size
              )

    rank = rhyme.predict([x_test0, x_test1])

    rank = rank.reshape(test_num, 300)
    rank_list = GR.get_rank_matrix(rank)
    result = GR.get_result_by_ranks(rank_list, rec_k_list)

    print('=======Rhyme2vec Result=======' + '\n')
    K.clear_session()
    return result, rhyme


def doc2vec(alpha=0, beta=0, activation=activ, use_bias=True, latent_dim=0, epochs=5, batch_size=100000):
    # recall@k
    rec_k_list = [1, 5, 30, 150]
    # the input dimension
    input_shape = doc_dim

    # Model
    x = Input(shape=(input_shape,))
    x_encoder = Dense(units=latent_dim, activation=activation, use_bias=use_bias)(x)
    sig = Dense(1, activation='sigmoid', use_bias=use_bias)(x_encoder)

    doc = Model(outputs=sig, inputs=x)
    print('=======Model Information=======' + '\n')
    doc.summary()
    doc.compile(optimizer=optim, loss='binary_crossentropy')

    doc.fit(x_train_doc, y_train,
            shuffle=False,
            epochs=epochs,
            batch_size=batch_size
            )

    rank = doc.predict(x_test_doc)

    rank = rank.reshape(test_num, 300)
    rank_list = GR.get_rank_matrix(rank)
    result = GR.get_result_by_ranks(rank_list, rec_k_list)
    print('=======Doc2vec Result=======' + '\n')
    K.clear_session()
    return result, doc


def con(alpha=0, beta=0, activation=activ, use_bias=True, latent_dim=0, epochs=5, batch_size=50000):
    # recall@k
    rec_k_list = [1, 5, 30, 150]

    # Model
    x_doc = Input(shape=(doc_dim,))
    x_0 = Input(shape=(pho_dim,))
    x_1 = Input(shape=(pho_dim,))
    x_pho, att = get_weighted([x_0, x_1], pho_dim)
    encoder = concatenate([x_doc, x_pho])
    encoder = Dense(units=latent_dim, activation=activation, use_bias=use_bias)(encoder)
    sig = Dense(1, activation='sigmoid', use_bias=use_bias)(encoder)
    con = Model(outputs=sig, inputs=[x_doc, x_0, x_1])
    print('=======Model Information=======' + '\n')
    con.summary()
    con.compile(optimizer=optim, loss='binary_crossentropy')
    con.fit([x_train_doc, x_train0, x_train1], y_train,
            shuffle=False,
            epochs=epochs,
            batch_size=batch_size
            )

    rank = con.predict([x_test_doc, x_test0, x_test1])
    rank = rank.reshape(test_num, 300)
    rank_list = GR.get_rank_matrix(rank)
    result = GR.get_result_by_ranks(rank_list, rec_k_list)

    print('=======Concatenate Result=======' + '\n')
    K.clear_session()
    return result, con

def attaerl(alpha=2, beta=1, gamma=1, delta=0, activation=activ, use_bias=False, epochs=20, batch_size=50000,
         units=128, latent_dim=100):
    # recall@k
    rec_k_list = [1, 5, 30, 150]

    # Input
    x_doc = Input(shape=(doc_dim,))
    x_0 = Input(shape=(pho_dim,))
    x_1 = Input(shape=(pho_dim,))
    x_pho, pho_att = get_weighted([x_0, x_1], pho_dim)

    doc_encoder = Dense(units=units, activation=activation, use_bias=use_bias)(x_doc)
    pho_encoder = Dense(units=units, activation=activation, use_bias=use_bias)(x_pho)

    # Attention Model
    u, u_att = get_weighted([doc_encoder, pho_encoder], units)

    v = Dense(units=latent_dim, activation=activation, use_bias=use_bias)(u)

    _u = Dense(units=units, activation=activation, use_bias=use_bias)(v)
    _doc_decoder, _pho_decoder = de_attention([_u, u_att], units, 2)

    _x_doc = Dense(units=doc_dim, activation=activation, use_bias=use_bias)(_doc_decoder)
    _x_pho = Dense(units=pho_dim, activation=activation, use_bias=use_bias)(_pho_decoder)

    _x_0, _x_1 = de_attention([_x_pho, pho_att], pho_dim, 2)

    y = Input(shape=(1,), name='y_in')
    sig = Dense(1, activation='sigmoid', use_bias=use_bias)(v)

    # Label loss
    def loss(args):
        x, y = args
        loss = objectives.binary_crossentropy(x, y)
        return loss

    def ae_loss(args):
        x, y = args
        loss = objectives.mean_squared_error(x, y)
        return loss

    label_loss = Lambda(loss)([y, sig])

    # Vae loss
    x_doc_loss = Lambda(ae_loss)([x_doc, _x_doc])
    x_0_loss = Lambda(ae_loss)([x_0, _x_0])
    x_1_loss = Lambda(ae_loss)([x_1, _x_1])


    L = CustomVariationalLayer(paras=[alpha, beta, gamma, delta])([label_loss, x_doc_loss, x_0_loss, x_1_loss])

    aerl = Model(outputs=L, inputs=[x_doc, x_0, x_1, y])
    print('=======Model Information=======' + '\n')
    aerl.summary()

    aerl.compile(optimizer=optim, loss=None)

    aerl.fit_generator(generate_batch_data_random([x_train_doc, x_train0, x_train1], y_train, batch_size),
                       steps_per_epoch=(y_train.shape[0] + batch_size - 1) //
                                       batch_size,
                       epochs=epochs,
                       verbose=1)

    aerl_sig = Model(inputs=[x_doc, x_0, x_1, y], outputs=sig)

    y_test = np.array(([1] + ([0] * 299)) * test_num)
    rank = aerl_sig.predict([x_test_doc, x_test0, x_test1, y_test])

    scores = rank.reshape(test_num, 300)
    rank_list = GR.get_rank_matrix(scores)
    result = GR.get_result_by_ranks(rank_list, rec_k_list)

    print('=======attaerl Result=======' + '\n')
    # K.clear_session()
    return result, aerl


def func(model, name, epoch, log_f, turns, dim):
    r = []
    log_f.write('\n======{}======\n'.format(name))
    for j in range(turns):
        result, m_f = model(alpha=1, beta=1, epochs=epoch, latent_dim=dim)
        line = '{}:> {}\n'.format(j, ' '.join([str(f1) for f1 in result]))
        log_f.write(line)
        r.append(result)
        print(result)
        log_f.flush()
        K.clear_session()
    t = np.mean(r, axis=0)
    print('')
    print('====== Average ======')
    print(t)
    log_f.write('Average: {}\n'.format(' '.join([str(fl) for fl in t])))
    log_f.write('======{} end.======\n'.format(name))
    log_f.flush()
    return t


if __name__ == '__main__':
    import time

    log = open('log', 'a')
    log.write("\n{}\n"
              "test size:{}\n".format(time.asctime(time.localtime(time.time())), test_num))

    # dims
    models = [doc2vec, rhyme2vec, con, attaerl]
    names = ['doc2vec', 'rhyme2vec', 'con', 'attaerl']
    turn = [10] * 4
    res = []
    dims = [100]
    times = []
    iset = [2]
    '''
    0: doc2vec
    1: rhyme2vec
    2: con
    3: attaerl
    '''
    print('================Dimension Discussion=================')
    for i in iset:
        print("\n\n*****************{}*****************\n\n".format(names[i]))
        for d in dims:
            time_start = time.time()
            res_t = func(models[i], names[i], 20, log, turn[i], d)
            time_used = time.time() - time_start
            log.write('{} dims take {} seconds in average.\n'.format(d, time_used))
            log.write('result: {}\n'.format(res_t))
            log.flush()
    log.close()
