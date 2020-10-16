import numpy as np
def sequence_length(trace, index, packet_dir):
    seq = 0
    while(index < len(trace)):
        if(trace[index] == packet_dir):
            seq += 1
        elif(trace[index] == 0):
            index = len(trace)
            break
        else:
            break
        index += 1
    return (seq, index)

def calc_dist_r(source, target, return_diff_and_idx = False):
    allowed_values = [-1, 0, 1]
    assert all([k in allowed_values for k in source])
    assert all([k in allowed_values for k in target])

    packet_dir = 1
    source_index = 0
    target_index = 0
    sqrd_sum = 0
    difference_amplifier = 30
    difference_accumulator = []
    source_indices = [0]
    
    while(source_index<len(source) and target_index<len(target)):

        src_seq_len, source_index = sequence_length(source, source_index, packet_dir)
        trgt_seq_len, target_index = sequence_length(target, target_index, packet_dir)

        packet_dir = -packet_dir

        difference = trgt_seq_len - src_seq_len
        sqrd_sum += np.square(difference)
        difference_accumulator.append(np.abs(difference_amplifier * difference))
        source_indices.append(source_index)
    
    distance = np.sqrt(sqrd_sum)
    if return_diff_and_idx:
        return (distance, difference_accumulator, source_indices)
    return distance
###########################################################
# NOTE THAT EVEN IF SOURCE AND TARGET LENGTHS ARE DIFFER- #
# ENT IT STILL GOES TILL THE END TO CALCULATE DISTANCE    #
###########################################################


#############################################################
# FROM THE BIG LOOP                                         #
#############################################################
def packet_count(trace):
    assert len(trace) == 5000 # BECAUSE DATA IS FROM DEEP FINGERPRINTING LITERATURE
    for p_c in range(5000):
        if trace[p_c] == 0:
            break
    return p_c

def cliptrace(trace):
    '''Clip trace so that it begins from a 1 direction.
    and pad accordingly.'''
    index = 0
    while (index < len(trace)):
        if trace[index] == 1:
            break
        index += 1
        ###################################################
        # FIND FIRST INDEX OF 1. POSSIBLY IMPLYING THAT   #
        # THE GUARD RELAY WON'T LEAK ANY TRACES. BUT THAT #
        # IS NOT POSSIBLE(?)                              #
        ###################################################
    trace = trace[index:]
    ###################################################
    # START TRACE FROM THERE AND IGNORE WHAT'S BEFORE #
    ###################################################

    trace = pad_to_5000(trace)
    return trace

def find_closest(Xtarget_pool, dummy_x):
    distmin = 1000000000
    closest = 0
    for Xindex in range(len(Xtarget_pool)):
        dist = calc_dist_r( Xtarget_pool[Xindex], dummy_x)
        if dist < distmin:
            distmin = dist
            closest = Xindex
    return closest

def pad_to_5000(trace):
    length = len(trace)
    for _ in range(length, 5000):
        trace.append(0)
    return trace
