from unet3d.utils.utils import pickle_dump, pickle_load

def generate_cross_val_fold():
    # generate 5-fold cross validation list, 1st one is list_validation_ids.pkl
    trained_list = pickle_load('../data/list_training_ids.pkl')
    val_list = pickle_load('../data/list_validation_ids.pkl')
    num_trained_list = len(trained_list)
    
    pickle_dump(trained_list[:int(0.25 * num_trained_list)],'../data/list_cv1_val.pkl')
    pickle_dump(val_list + trained_list[int(0.25 * num_trained_list):],'../data/list_cv1_train.pkl')
    
    pickle_dump(trained_list[int(0.25 * num_trained_list):int(0.5 * num_trained_list)],'../data/list_cv2_val.pkl')
    pickle_dump(val_list + trained_list[:int(0.25 * num_trained_list)] + trained_list[int(0.5 * num_trained_list):],
               '../data/list_cv2_train.pkl')
    
    pickle_dump(trained_list[int(0.5 * num_trained_list):int(0.75 * num_trained_list)],'../data/list_cv3_val.pkl')
    pickle_dump(val_list + trained_list[:int(0.5 * num_trained_list)] + trained_list[int(0.75 * num_trained_list):],
               '../data/list_cv3_train.pkl')
    
    pickle_dump(trained_list[int(0.75 * num_trained_list):],'../data/list_cv4_val.pkl')
    pickle_dump(val_list + trained_list[:int(0.75 * num_trained_list)],'../data/list_cv4_train.pkl')
    
    