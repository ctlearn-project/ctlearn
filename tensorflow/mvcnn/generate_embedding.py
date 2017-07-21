
def generateEmbedding():
    #open file to determine the telescopes included
    f = open_file(data_file, mode = "r", title = "Input file")
    table = f.root.E0.Events_Training
    table_val = f.root.E0.Events_Validation
    columns_list = table.colnames 
    tels_list = []
    for i in columns_list:
        if re.match("T[0-9]+",i):
            tels_list.append(i)

    num_tel = len(tels_list)

    num_events_training = table.shape[0]
    num_events_val = table_val.shape[0]
    batch_indices = [i for i in range(num_events_training) if i%BATCH_SIZE == 0 and i+BATCH_SIZE < num_events_training]
    batches_per_epoch = len(batch_indices)

    print("Batch size: ",BATCH_SIZE)
    print("Total number of training events: ",num_events_training)
    print("Total number of validation events: ",num_events_val)
    print("Batches per epoch: ",batches_per_epoch) 

if __name__ == '__main__':
    
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Takes a logdir, loads the model, then generates an embedding and')
    parser.add_argument('h5_file', help='path to h5 file containing data')
    parser.add_argument('--epochs',default=10000)
    parser.add_argument('--logdir',default='runs/mvcnn1')
    parser.add_argument('--lr',default=0.001)
    parser.add_argument('--checkpoint_basename',default='mvcnn.ckpt')
    args = parser.parse_args()

    train(args.h5_file,args.epochs)
