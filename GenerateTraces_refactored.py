import math
import pickle
import random
import numpy as np
from Model_NoDef import DFNet
from keras.utils import np_utils
from keras.optimizers import Adamax
from keras.models import load_model
import tensorflow as tf

from util import calc_dist_r
from util import cliptrace, pad_to_5000, find_closest, packet_count

# FOR REPRODUCIBILITY
random.seed(1)
np.random.seed(1)
np.random.RandomState(1)
tf.random.set_seed(1)

def generate_target_pools(l):
    global X_test, y_test, size
    X_pool = []
    y_pool = []
    ctr = 0
    while (ctr < 100):
        index=random.randrange(size)
        if (index != l):
            ctr += 1
            X_pool.append(X_test[index])
            y_pool.append(y_test[index])
    return (X_pool, y_pool)


def create_model():
    LENGTH = 5000 # Packet sequence length
    INPUT_SHAPE = (LENGTH,1)
    NB_CLASSES = 95 # number of outputs = number of classes
    OPTIMIZER = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) # Optimizer

    model = DFNet.build(input_shape=INPUT_SHAPE, classes=NB_CLASSES)
    model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER,metrics=["accuracy"])
    return model

def converge_to_target(source, target, model, val, dummy_y):
    answer = source[:] # ANSWER IS INIT TO SOURCE

    isConverged = 0
    
    for x in range (10):
        total, arr, index = calc_dist_r(source, target, True)
        value = 0
        
        for y in range(len(arr)):
            for i in range(int(arr[y]/total)):
                answer.insert(value+index[y+1],1)
            value += abs(int(arr[y]/total)) # INCREASE VALUE BY ARR[y]/TOTAL WHERE ARR[y] IS 30*DIS
            # WHAT DOES ARR[y] / TOTAL REPRESENT?
    
        padded_answer = answer[:]
        padded_answer = pad_to_5000(padded_answer)

        padded_answer=padded_answer[:5000]
        unreshaped_answer=padded_answer[:]

        padded_answer=np.array(padded_answer)
        padded_answer=np.array([padded_answer])

        padded_answer=padded_answer.astype('float32')
        padded_answer=padded_answer[:,:,np.newaxis]
        ###################################################
        # PROPER RESHAPING OF ARRAY.                      #
        ###################################################
        
        predict = model.predict_classes(padded_answer)
        prediction = int(predict[0])

        if(val == prediction or dummy_y != prediction):
            isConverged = 1
            break
        source = answer
    return (isConverged, unreshaped_answer, prediction, value)


with open('/content/df-master/dataset/ClosedWorld/NoDef/X_test_NoDef.pkl', 'rb') as handle:
        X_test = np.array(pickle.load(handle,encoding="bytes"))
with open('/content/df-master/dataset/ClosedWorld/NoDef/y_test_NoDef.pkl', 'rb') as handle:
        y_test = np.array(pickle.load(handle,encoding="bytes"))
size=X_test.shape[0]

def main():
    global X_test, y_test, size
    VERBOSE = 2 # Output display mode
    NB_CLASSES = 95 # number of outputs = number of classes

    '''
    outfile1 = open("/content/Adversarial_Traces_x_alpha30_only1.pkl",'wb')
    outfile2 = open("/content/Adversarial_Traces_y_alpha30_only1.pkl",'wb')'''

    datax=[]
    datay=[]
    model=create_model()
    totalpadding=0
    packets=0
    model.load_weights('/content/torsheild/nodef_model_weights_trainer.h5')

    for l in range(0,9500):
        dummy_x = X_test[l]
        dummy_y = y_test[l]
        source = dummy_x.tolist()
        packetcount = packet_count(source)
        for t in range(0,100):
            ###################################################
            # MONTE-CARLO LIKE RANDOMLY CONVERGING TO A TARGET#
            ###################################################
            Xtarget_pool, ytarget_pool = generate_target_pools(l)

            Xtarget_pool = np.array(Xtarget_pool)
            ytarget_pool = np.array(ytarget_pool)
            
            source = cliptrace(source)

            dummy_x = source[:]
            
            for X_index in range(len(Xtarget_pool)):
                target = Xtarget_pool[X_index].tolist()
                target = cliptrace(target)
                Xtarget_pool[X_index] = target[:]
            
            closest = find_closest(Xtarget_pool, dummy_x)

            val = ytarget_pool[closest]
            target = Xtarget_pool[closest].tolist()
            
            Xtarget_pool = Xtarget_pool.astype('float32')
            ytarget_pool = ytarget_pool.astype('float32')
            
            Xtarget_pool = Xtarget_pool[:,:,np.newaxis]
            ytarget_pool = np_utils.to_categorical(ytarget_pool, NB_CLASSES)
            score_test = model.evaluate(Xtarget_pool, ytarget_pool, verbose=VERBOSE)

            isConverged, obfuscated_source, prediction, value = converge_to_target(source, target, model, val, dummy_y)
            
            if(isConverged):
                break
        datax.append(np.array(obfuscated_source))
        ##########################################################
        # WHEN LOOP IS EXITED, APPEND OBFUSCATED SOURCE IN DATAX #
        # BUT APPEND PREDICTION IN DATAY? WHAT?                  #
        ##########################################################
        datay.append(prediction)
        totalpadding = totalpadding + value
        packets = packets + packetcount
        if (l % 50 == 0):
            print("For ",l,"Class is ",value)

    print("Total padding ",totalpadding)
    print("total packets",packets)
    print("percentage %.2f%%"%((totalpadding/(totalpadding+packets))*100))

    '''pickle.dump(datax,outfile1)
    pickle.dump(datay,outfile2)

    outfile1.close()
    outfile2.close()'''

if __name__=="__main__":
    main()
