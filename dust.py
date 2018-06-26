import numpy as np
import tensorflow as tf
from random import randint
import sys

# execution: python3 dust.py 111121
def renormalization(df):
    df_norm=df
    d1,d2=np.shape(df)
    for i in range(d2):
        df_norm[:,i:i+1]=np.divide(df[:,i:i+1]-np.amin(df[:,i:i+1]),np.amax(df[:,i:i+1])-np.amin(df[:,i:i+1]))
        if i > d2-3:
            print(i,np.amax(df[:,i:i+1]),np.amin(df[:,i:i+1]))
    return df_norm 


def TD_CNN(df,timewindow,location):
    f = open('model2_tdcnn_log_'+str(timewindow)+"_"+location+'.txt', 'w')
    s1,s2=np.shape(df) #(24029,243)
    # Generate input by dividing dataset with time period timewindow(48)
#    input_set=[]; output_set=[];
#    for i in range(s1-timewindow):
#        input_set.append(df[i:i+timewindow,0:s2-2]) #using input[t=0:t=47] for 48 hrs: 2 days
#        output_set.append(df[i+timewindow-1,s2-2:s2]) #predict output [t=47]
#    # shuffle time order
#    index_list=[i for i in range(len(input_set))]
#    input_set_shuffle=[]; output_set_shuffle=[];
#    for i in index_list:
#        input_set_shuffle.append(input_set[i])
#        output_set_shuffle.append(output_set[i])
#    input_set_shuffle=np.asarray(input_set_shuffle) #(23981, 48, 243) 
#    output_set_shuffle=np.asarray(output_set_shuffle) #(23981, 2)
#    print(np.shape(input_set_shuffle),np.shape(output_set_shuffle))
#    d1,d2,d3=np.shape(input_set_shuffle)
#    input_train=input_set_shuffle[0:int(d1*0.8),:,:]
#    input_test =input_set_shuffle[int(d1*0.8):d1,:,:]
#    output_train=output_set_shuffle[0:int(d1*0.8),:]
#    output_test=output_set_shuffle[int(d1*0.8):d1,:]
#    x=tf.placeholder(tf.float32, shape=[None,d2,d3])
#    y_=tf.placeholder(tf.float32, shape=[None,2])
#    keep_prob = tf.placeholder(tf.float32)

    input_set=[]; output_set=[];
    for i in range(s1-timewindow):
        input_set.append(df[i:i+timewindow,0:s2-2]) #using input[t=0:t=47] for 48 hrs: 2 days
        output_set.append(df[i+timewindow-1,s2-2:s2]) #predict output [t=47]
    input_test =np.asarray(input_set[int(len(input_set)*0.8):len(input_set)])
    output_test=np.asarray(output_set[int(len(input_set)*0.8):len(input_set)])
    # shuffle time order
    index_list=[i for i in range(int(len(input_set)*0.8))]
    input_set_shuffle=[]; output_set_shuffle=[];
    for i in index_list:
        input_set_shuffle.append(input_set[i])
        output_set_shuffle.append(output_set[i])
    input_train=np.asarray(input_set_shuffle) #(23981, 48, 243) 
    output_train=np.asarray(output_set_shuffle) #(23981, 2)
    print(np.shape(input_set_shuffle),np.shape(output_set_shuffle))
    d1,d2,d3=np.shape(input_set_shuffle)


    #TD_CNN (2D CNN) model
    input_layer = tf.reshape(x, [-1, d2, d3, 1])
    conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=64,
      kernel_size=[8, 243],
      padding="same",
      activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 3], strides=2)
    conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=128,
      kernel_size=[8, 60],
      padding="same",
      activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 3], strides=2)
    pool2_flat = tf.reshape(pool2, [-1, int(timewindow/4) * int(d3/3/3) * 128])
    dense1 = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout1 = tf.layers.dropout(
          inputs=dense1, rate=keep_prob)
    dense2 = tf.layers.dense(inputs=dropout1, units=512, activation=tf.nn.relu)
    dropout2 = tf.layers.dropout(
          inputs=dense2, rate=keep_prob)
    prediction = tf.layers.dense(inputs=dropout2, units=2)
    loss = tf.losses.mean_squared_error(y_, prediction)
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    #Training and Testing 
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    for i in range(50000):
      #obtain random batch for training
      tr_input_batch=[]; tr_output_batch=[];
      #generating mini batch with size of 1000
      batch_size=500
      for j in range(batch_size):
          index=randint(0,int(d1*0.8)-1)
          tr_input_batch.append(np.reshape(input_train[index,:,:],[1,d2,d3]))
          tr_output_batch.append(np.reshape(output_train[index,:],[1,2]))
      tr_input=np.concatenate(tr_input_batch,0)
      tr_output=np.concatenate(tr_output_batch,0)
      #start train
      sess.run(train_op,feed_dict={x: tr_input, y_:tr_output, keep_prob: 0.5})
      if i % 10 == 0:
        train_loss = sess.run(loss,feed_dict={
            x: tr_input, y_:tr_output, keep_prob: 1.0})
        print('step %d, training loss (MSE) %g' % (i, 100.0*train_loss))
        f.write('step %d, training loss (MSE) %g' % (i, 100.0*train_loss))
      #start train
      if i%100 ==0:
        print('step %d test loss (MSE) %g \n' % (i, sess.run(loss*100.0,feed_dict={
        x:input_test , y_: output_test, keep_prob: 1.0})))
        #use all test data for checking testing loss
        #It is cheating but just check the testing loss in this case with lack of training data
        f.write('step %d test loss (MSE)  %g \n' % (i, sess.run(loss*100.0,feed_dict={
        x:input_test , y_: output_test, keep_prob: 1.0})))
        prediction_val=sess.run(prediction,feed_dict={
                x:input_test , y_: output_test, init_state: zero_state_te})
        np.save("td_cnn_pr_"+str(timewindow)+"_"+location+".npy",prediction_val)
        np.save("td_cnn_gt_"+str(timewindow)+"_"+location+".npy",output_test)

    f.close()



#function for LSTM model
def lstm_cell(lstm_size):
  return tf.nn.rnn_cell.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(lstm_size),output_keep_prob=0.5)

def LSTM(df,timewindow,location):
    f = open('model3_lstm_log_'+str(timewindow)+"_"+location+'.txt', 'w')
    s1,s2=np.shape(df)
    # Generate input by dividing dataset with time period 40
    input_set=[]; output_set=[];
    for i in range(s1-timewindow):
        input_set.append(df[i:i+timewindow,0:s2-2]) #using input[t=0:t=47] for 48 hrs: 2 days
        output_set.append(df[i+timewindow-1,s2-2:s2]) #predict output [t=47]
    input_test =np.asarray(input_set[int(len(input_set)*0.8):len(input_set)])
    output_test=np.asarray(output_set[int(len(input_set)*0.8):len(input_set)])
    # shuffle time order
    index_list=[i for i in range(int(len(input_set)*0.8))]
    input_set_shuffle=[]; output_set_shuffle=[];
    for i in index_list:
        input_set_shuffle.append(input_set[i])
        output_set_shuffle.append(output_set[i])
    input_train=np.asarray(input_set_shuffle) #(23981, 48, 243) 
    output_train=np.asarray(output_set_shuffle) #(23981, 2)
    print(np.shape(input_set_shuffle),np.shape(output_set_shuffle))
    d1,d2,d3=np.shape(input_set_shuffle)
    #LSTM Model
    timestep=timewindow; #(48)
    number_of_layers=3; lstm_size=42; batch_size=500;
    init_state=tf.placeholder(tf.float32, [number_of_layers, 2, None, lstm_size])
    x=tf.placeholder(tf.float32, [None,timestep,243])
    y_=tf.placeholder(tf.float32, [None,2])
    stacked_lstm=tf.contrib.rnn.MultiRNNCell(
        [lstm_cell(lstm_size) for _ in range(number_of_layers)])
    state_per_layer_list = tf.unstack(init_state, axis=0)
    rnn_tuple_state = tuple(
        [tf.nn.rnn_cell.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
         for idx in range(number_of_layers)]
    )
    weights = {
        'out': tf.Variable(tf.random_normal([lstm_size, 2]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([2]))
    }
    outputs, last_states = tf.nn.dynamic_rnn(cell=stacked_lstm, inputs=x , sequence_length=None, dtype=tf.float32, initial_state=rnn_tuple_state)
    t1,t2,t3=outputs.get_shape().as_list() #(batch_size, timestep, lstm_size)
    outputs = tf.reshape(outputs[:,t2-1,:], [-1,t3]) #select last timestep only
    out=tf.matmul(outputs, weights['out']) + biases['out'];
    prediction=tf.reshape(out, [ -1, 2]) #(batch_size,2)
    last_states=tf.stack(last_states,axis=0)
    loss = tf.losses.mean_squared_error(y_, prediction)
    optimizer = tf.train.AdamOptimizer(learning_rate=5e-4)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    #Training and Testing 
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    for i in range(500000):
        zero_state=np.zeros((number_of_layers, 2, batch_size, lstm_size))
        #obtain random batch for training
        tr_input_batch=[]; tr_output_batch=[];
        #generating mini batch with size of 1000
        for j in range(batch_size):
            index=randint(0,int(d1*0.8)-1)
            tr_input_batch.append(np.reshape(input_train[index,:,:],[1,timestep,243]))
            tr_output_batch.append(np.reshape(output_train[index,:],[1,2]))
        tr_input=np.concatenate(tr_input_batch,0)
        tr_output=np.concatenate(tr_output_batch,0)
        #start train
        sess.run(train_op,feed_dict={x: tr_input, y_:tr_output, init_state: zero_state})
        if i % 10 == 0:
            train_loss = sess.run(loss,feed_dict={
              x: tr_input, y_:tr_output, init_state: zero_state})
            print('step %d, training loss (MSE) %g' % (i, 100.0*train_loss))
            f.write('step %d, training loss (MSE) %g\n' % (i, 100.0*train_loss))
        #start test
        if i%100 ==0:
            zero_state_te=np.zeros((number_of_layers, 2, len(input_test), lstm_size))
            print('step %d test loss (MSE) %g \n' % (i, sess.run(loss*100.0,feed_dict={
                x:input_test , y_: output_test, init_state: zero_state_te})))
            #use all test data for checking testing loss
            #It is cheating but just check the testing loss in this case with lack of training data
            f.write('step %d test loss (MSE)  %g \n' % (i, sess.run(loss*100.0,feed_dict={
                x:input_test , y_: output_test, init_state: zero_state_te})))
            prediction_val=sess.run(prediction,feed_dict={
                x:input_test , y_: output_test, init_state: zero_state_te})
            np.save("lstm_pr_"+str(timewindow)+"_"+location+".npy",prediction_val)
            np.save("lstm_gt_"+str(timewindow)+"_"+location+".npy",output_test)
    f.close()



def main():
    location=sys.argv[1]
    y=np.load("Label_"+location+".npy")
    x=np.load("All_input.npy")
    df=np.concatenate([x,y],1)
    #Renormalization
    df=renormalization(df)
    print(np.shape(df))
    print(df)
    ## Model 2 - TD CNN
    #TD_CNN(df,48,location) #timewindow=48
    ## Model 3 - LSTM
    LSTM(df,48,location)

if __name__ == '__main__':
    main()
