import tensorflow as tf
import numpy as np
import os
import time
from utils import random_batch, normalize, similarity, loss_cal, optim, factory_input, tsne_plot
from configuration import get_config
from tensorflow.contrib import rnn
import shutil

config = get_config()


def train(path):
    tf.reset_default_graph()    # reset graph


    # draw graph
    batch = tf.placeholder(shape= [None, config.N*config.M, 40], dtype=tf.float32)  # input batch (time x batch x n_mel)
    lr = tf.placeholder(dtype= tf.float32)  # learning rate
    global_step = tf.Variable(0, name='global_step', trainable=False)
    w = tf.get_variable("w", initializer= np.array([10], dtype=np.float32))
    b = tf.get_variable("b", initializer= np.array([-5], dtype=np.float32))

    # embedding lstm (3-layer default)
    with tf.variable_scope("lstm"):
        lstm_cells = [tf.contrib.rnn.LSTMCell(num_units=config.hidden, num_proj=config.proj) for i in range(config.num_layer)]
        lstm = tf.contrib.rnn.MultiRNNCell(lstm_cells)    # define lstm op and variables
        outputs, _ = tf.nn.dynamic_rnn(cell=lstm, inputs=batch, dtype=tf.float32, time_major=True)   # for TI-VS must use dynamic rnn
        embedded = outputs[-1]                            # the last ouput is the embedded d-vector
        embedded = normalize(embedded)                    # normalize
    print("embedded size: ", embedded.shape)

    # loss
    sim_matrix = similarity(embedded, w, b)
    print("similarity matrix size: ", sim_matrix.shape)
    loss = loss_cal(sim_matrix, type=config.loss)

    # optimizer operation
    trainable_vars= tf.trainable_variables()                # get variable list
    optimizer= optim(lr)                                    # get optimizer (type is determined by configuration)
    grads, vars= zip(*optimizer.compute_gradients(loss))    # compute gradients of variables with respect to loss
    grads_clip, _ = tf.clip_by_global_norm(grads, 3.0)      # l2 norm clipping by 3
    grads_rescale= [0.01*grad for grad in grads_clip[:2]] + grads_clip[2:]   # smaller gradient scale for w, b
    train_op= optimizer.apply_gradients(zip(grads_rescale, vars), global_step= global_step)   # gradient update operation

    # check variables memory
    variable_count = np.sum(np.array([np.prod(np.array(v.get_shape().as_list())) for v in trainable_vars]))
    print("total variables :", variable_count)

    # record loss
    loss_summary = tf.summary.scalar("loss", loss)
    merged = tf.summary.merge_all()
    saver = tf.train.Saver()
    iter = 0



    # training session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        if config.restore:

                # Restore saved model if the user requested it, default = True
            try:
                ckpt = tf.train.latest_checkpoint(checkpoint_dir=os.path.join(path,"Check_Point"))

#                if (checkpoint_state and checkpoint_state.model_checkpoint_path):
#                    print('Loading checkpoint {}'.format(checkpoint_state.model_checkpoint_path))
            #saver = tf.train.import_meta_graph(os.path.join(path,"Check_Point/model.cpkt.meta"))

            #ckpt = tf.train.load_checkpoint(os.path.join(path,"Check_Point/model"))
                saver.restore(sess, ckpt)


#                else:
#                    print('No model to load at {}'.format(save_dir))

#                    saver.save(sess, checkpoint_path, global_step=global_step)


            except:
                print('Cannot restore checkpoint exception')


        #if loaded == 0:
        #    raise AssertionError("ckpt file does not exist! Check config.model_num or config.model_path.")

        #print("train file path : ", config.test_path)




        else:

            os.makedirs(os.path.join(path, "Check_Point"), exist_ok=True)  # make folder to save model
            os.makedirs(os.path.join(path, "logs"), exist_ok=True)          # make folder to save log

        writer = tf.summary.FileWriter(os.path.join(path, "logs"), sess.graph)
        epoch = 0
        lr_factor = 1   # lr decay factor ( 1/2 per 10000 iteration)
        loss_acc = 0    # accumulated loss ( for running average of loss)
        iter=0
        training_data_size = len(os.listdir(config.train_path))
        print("train_size: " , training_data_size)
        prev_iter = -1




        #        while iter  < config.iteration :
        while iter < config.iteration:
            prev_iter = iter

            # run forward and backward propagation and update parameters
            iter, _ ,loss_cur, summary = sess.run([global_step,train_op, loss, merged],
                                  feed_dict={batch: random_batch(), lr: config.lr*lr_factor})

            loss_acc += loss_cur    # accumulated loss for each 100 iteration


            if(iter - prev_iter > 1):
                epoch = config.N * (iter+1) // training_data_size
                #lr_factor = lr_factor / (2**(epoch//100))
                lr_factor = lr_factor / (2**(iter//10000))
                print("restored epoch:", epoch)
                print("restored learning rate:", lr_factor*config.lr)



            #if iter % 1000 == 0:
            #    writer.add_summary(summary, iter)   # write at tensorboard
            if (iter+1) % 100 == 0:
                print("(iter : %d) loss: %.4f" % ((iter+1),loss_acc/100))
                loss_acc = 0                        # reset accumulated loss

            #if config.N * (iter+1) % training_data_size == 0:
            #    epoch = epoch + 1
            #    print("epoch: ", epoch)
                
            if (iter+1) % 10000 == 0:
                lr_factor /= 2
                print("learning rate is decayed! current lr : ", config.lr*lr_factor)
                
            




            #if ((config.N * (iter+1)) / training_data_size)%100  == 0:
            #    lr_factor = lr_factor / 2
            #    print("learning factor: " , lr_factor)
            #    print("learning rate is decayed! current lr : ", config.lr*lr_factor)

            if (iter+1) % 5000 == 0:
                saver.save(sess, os.path.join(path, "Check_Point/model.ckpt"), global_step=iter) #pooooooooooooint
                writer.add_summary(summary, iter)   # write at tensorboard
                print("model is saved!")

            #if (iter+1) % 10000 == 0:
            #    os.mkdir(os.path.join(config.gdrive_path, "speaker_vertification_model_vox_"+str(iter+1)))
            #    shutil.rmtree(os.path.join(config.gdrive_path, "speaker_vertification_model_vox_"+str(iter-10000+1)))

            #    shutil.copytree(path,os.path.join(config.gdrive_path, "speaker_vertification_model_vox_"+str(iter+1)))









# Test Session
def test(path):
    tf.reset_default_graph()

    # draw graph
    enroll = tf.placeholder(shape=[None, config.N*config.M, 40], dtype=tf.float32) # enrollment batch (time x batch x n_mel)
    verif = tf.placeholder(shape=[None, config.N*config.M, 40], dtype=tf.float32)  # verification batch (time x batch x n_mel)
    batch = tf.concat([enroll, verif], axis=1)

    # embedding lstm (3-layer default)
    with tf.variable_scope("lstm"):
        lstm_cells = [tf.contrib.rnn.LSTMCell(num_units=config.hidden, num_proj=config.proj) for i in range(config.num_layer)]
        lstm = tf.contrib.rnn.MultiRNNCell(lstm_cells)    # make lstm op and variables
        outputs, _ = tf.nn.dynamic_rnn(cell=lstm, inputs=batch, dtype=tf.float32, time_major=True)   # for TI-VS must use dynamic rnn
        embedded = outputs[-1]                            # the last ouput is the embedded d-vector
        embedded = normalize(embedded)                    # normalize

    print("embedded size: ", embedded.shape)

    # enrollment embedded vectors (speaker model)
    enroll_embed = normalize(tf.reduce_mean(tf.reshape(embedded[:config.N*config.M, :], shape= [config.N, config.M, -1]), axis=1))
    # verification embedded vectors
    verif_embed = embedded[config.N*config.M:, :]

    similarity_matrix = similarity(embedded=verif_embed, w=1., b=0., center=enroll_embed)
    loss = loss_cal(similarity_matrix, type=config.loss)



    saver = tf.train.Saver(var_list=tf.global_variables())
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        # load model

        #ckpt = tf.train.get_checkpoint_state(path)
        #checkpoints =  ckpt.all_model_checkpoint_paths
        i=139999
        least_loss = 99999
        #print("checkpoints : ",checkpoints)

        while(i<399999):
            saver.restore(sess,os.path.join( path , "model.ckpt-"+str(i)))



            S, L = sess.run([similarity_matrix, loss], feed_dict={enroll:random_batch(shuffle=False),
                                                           verif:random_batch(shuffle=False, utter_start=config.M)})
            S = S.reshape([config.N, config.M, -1])
            print("test file path : ", config.test_path)
            np.set_printoptions(precision=2)
            #print(S)

            if L < least_loss:
                #diff = abs(FAR-FRR)
                perfect_step = i
                least_loss = L

            print(i)
            print(str(L/config.N))
            i = i + 5000


        print("\ncheckpoint: "+ str(perfect_step) + " (loss:%0.2f)"%(least_loss))




        #ckpt_list = ckpt.all_model_checkpoint_paths
        #loaded = 0
        #for model in ckpt_list:
        #    if config.model_num == int(model[-1]):    # find ckpt file which matches configuration model number
        #        print("ckpt file is loaded !", model)
        #        loaded = 1
        #        saver.restore(sess, model)  # restore variables from selected ckpt file
        #        break
        #print("checkpoint_directory:::::: ",ckpt)


        #if loaded == 0:
        #    raise AssertionError("ckpt file does not exist! Check config.model_num or config.model_path.")



        # return similarity matrix after enrollment and verification
        #time1 = time.time() # for check inference time
        #if config.tdsv:
        #    S = sess.run(similarity_matrix, feed_dict={enroll:random_batch(shuffle=False, noise_filenum=1),
                                                      # verif:random_batch(shuffle=False, noise_filenum=2)})
    #    else:

        #time2 = time.time()


        #print("inference time for %d utterences : %0.2fs"%(2*config.M*config.N, time2-time1))
            # print similarity matrix

        # calculating EER




def output(model_path):


    tf.reset_default_graph()

    N = len(os.listdir(config.test_path))

    # draw graph
    enroll = tf.placeholder(shape=[None, N*config.M, 40], dtype=tf.float32) # enrollment batch (time x batch x n_mel)
    #verif = tf.placeholder(shape=[None, config.N*config.M, 40], dtype=tf.float32)  # verification batch (time x batch x n_mel)
    #batch = tf.concat([enroll, verif], axis=1)

    # embedding lstm (3-layer default)
    with tf.variable_scope("lstm"):
        lstm_cells = [tf.contrib.rnn.LSTMCell(num_units=config.hidden, num_proj=config.proj) for i in range(config.num_layer)]
        lstm = tf.contrib.rnn.MultiRNNCell(lstm_cells)    # make lstm op and variables
        outputs, _ = tf.nn.dynamic_rnn(cell=lstm, inputs=enroll, dtype=tf.float32, time_major=True)   # for TI-VS must use dynamic rnn
        embedded = outputs[-1]                            # the last ouput is the embedded d-vector
        embedded = normalize(embedded)                    # normalize

    #print("embedded size: ", embedded.shape)

    # enrollment embedded vectors (speaker model)
    enroll_embed = normalize(tf.reduce_mean(tf.reshape(embedded[:N*config.M, :], shape= [N, config.M, -1]), axis=1))
    # verification embedded vectors
    #verif_embed = embedded[config.N*config.M:, :]

    #similarity_matrix = similarity(embedded=verif_embed, w=1., b=0., center=enroll_embed)

    saver = tf.train.Saver(var_list=tf.global_variables())
    with tf.Session() as sess:


        tf.global_variables_initializer().run()

        # load model
        print("model path :", model_path)
        #ckpt = tf.train.latest_checkpoint(checkpoint_dir=os.path.join(path, "Check_Point"))
        #ckpt = tf.train.latest_checkpoint(checkpoint_dir=model_path)
        #saver.restore(sess, ckpt)
        saver.restore(sess,os.path.join( model_path , "model.ckpt-"+str(config.restore_step)))

        #if loaded == 0:
        #    raise AssertionError("ckpt file does not exist! Check config.model_num or config.model_path.")

        print("test file path : ", config.test_path)

        # return similarity matrix after enrollment and verification
        time1 = time.time() # for check inference time
        if config.tdsv:
            S = sess.run(similarity_matrix, feed_dict={enroll:random_batch(shuffle=False, noise_filenum=1),
                                                       verif:random_batch(shuffle=False, noise_filenum=2)})
        else:
            e = sess.run(enroll_embed, feed_dict={enroll:factory_input()})

        print("embedding shape: " , e.shape)
        print("embedding: " , e)


    #np.save(embedding_file_name,e)


    #n = len(os.listdir(config.test_path))
    #speaker_dict = [None] * n
    #tsne_plot( os.listdir(config.test_path) , e )
    return e


    #dict_array = np.array(speaker_dict)
    #np.save("speakers_dictionary",dict_array)
def write_output_in_files(model_path):
    e = output(model_path)
    embedding_folder_name = "speaker_embeddings"
    for i, file in enumerate(os.listdir(config.test_path)):
        #speaker_dict[i] = file.strip('/')

        np.save("../"+embedding_folder_name+"/"+file,e[i])
